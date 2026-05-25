use std::fmt;

#[derive(Debug)]
pub enum BinderError {
    Io(std::io::Error),
    Parse(String),
    EmptyInput,
}

impl fmt::Display for BinderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinderError::Io(e) => write!(f, "I/O error: {e}"),
            BinderError::Parse(s) => write!(f, "Parse error: {s}"),
            BinderError::EmptyInput => write!(f, "No residues found in input"),
        }
    }
}

impl From<std::io::Error> for BinderError {
    fn from(e: std::io::Error) -> Self {
        BinderError::Io(e)
    }
}
