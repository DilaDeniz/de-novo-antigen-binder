/// wgpu GPU compute engine for population-level energy evaluation.
///
/// `GpuContext::try_init()` returns `None` if no suitable GPU adapter is found,
/// allowing the diffusion engine to fall back to the CPU Rayon path transparently.
///
/// Buffer layout (one `GpuAtom` = 32 bytes = 8 × f32):
///   { x, y, z, q, r_min_half, epsilon, hydrophobic_f32, _pad }
///
/// Shader file is embedded at compile time so the binary is self-contained.
#[cfg(feature = "gpu")]

use wgpu::util::DeviceExt;
use crate::allatom::AtomCloud;

// ── GpuAtom ───────────────────────────────────────────────────────────────────

/// Matches the `GpuAtom` struct in `shaders/energy.wgsl` exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAtom {
    pub x:          f32,
    pub y:          f32,
    pub z:          f32,
    pub q:          f32,
    pub r_min_half: f32,
    pub epsilon:    f32,
    pub hydrophobic: f32,
    pub _pad:       f32,
}

/// Uniforms buffer (must be 16-byte aligned for `var<uniform>`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n_ag:  u32,
    n_ab:  u32,
    _pad0: u32,
    _pad1: u32,
}

// Embed the shader at compile time
const SHADER_SRC: &str = include_str!("../shaders/energy.wgsl");

// ── GpuContext ────────────────────────────────────────────────────────────────

pub struct GpuContext {
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bg_layout: wgpu::BindGroupLayout,
}

impl GpuContext {
    /// Attempt to initialise a GPU context.  Returns `None` if no adapter found.
    pub fn try_init() -> Option<Self> {
        pollster::block_on(Self::try_init_async())
    }

    async fn try_init_async() -> Option<GpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("binder"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("energy"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                // antigen (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // antibodies (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // energies (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("energy"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Some(GpuContext { device, queue, pipeline, bg_layout })
    }

    // ── Batch energy evaluation ────────────────────────────────────────────────

    /// Score all candidates in a single GPU dispatch.
    ///
    /// `antigen`    — the target protein (all heavy atoms).
    /// `candidates` — flat slice: candidate 0 atoms first, candidate 1 next, …
    /// `n_ab`       — atoms per candidate (all candidates must have the same count).
    ///
    /// Returns one energy value per candidate (kcal/mol).
    pub fn score_batch(
        &self,
        antigen:    &AtomCloud,
        candidates: &[GpuAtom],  // flattened: n_cand × n_ab atoms
        n_ab:       usize,
    ) -> Vec<f32> {
        let n_ag   = antigen.len();
        let n_cand = if n_ab == 0 { 0 } else { candidates.len() / n_ab };
        if n_cand == 0 { return Vec::new(); }

        // ── Upload antigen ──────────────────────────────────────────────────────
        let ag_atoms: Vec<GpuAtom> = (0..n_ag)
            .map(|i| GpuAtom {
                x:           antigen.x[i],
                y:           antigen.y[i],
                z:           antigen.z[i],
                q:           antigen.charge[i],
                r_min_half:  antigen.r_min_half[i],
                epsilon:     antigen.epsilon[i],
                hydrophobic: antigen.hydrophobic[i] as f32,
                _pad:        0.0,
            })
            .collect();

        let ag_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("antigen"),
            contents: bytemuck::cast_slice(&ag_atoms),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let ab_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("antibodies"),
            contents: bytemuck::cast_slice(candidates),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let uniforms = Uniforms {
            n_ag:  n_ag  as u32,
            n_ab:  n_ab  as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniforms_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let energy_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("energies"),
            size:               (n_cand * std::mem::size_of::<f32>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("readback"),
            size:               (n_cand * std::mem::size_of::<f32>()) as u64,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("bg"),
            layout:  &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ab_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniforms_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: energy_buf.as_entire_binding() },
            ],
        });

        // ── Encode + submit ─────────────────────────────────────────────────────
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_cand as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &energy_buf, 0,
            &readback_buf, 0,
            (n_cand * std::mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(std::iter::once(enc.finish()));

        // ── Read back ───────────────────────────────────────────────────────────
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv().expect("GPU readback channel closed");

        let mapped = slice.get_mapped_range();
        let energies: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback_buf.unmap();

        energies
    }
}
