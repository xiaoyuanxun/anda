use object_store::path::DELIMITER;
use std::{future::Future, pin::Pin};

pub mod agent;
pub mod context;
pub mod http;
pub mod json;
pub mod model;
pub mod tool;

pub use agent::*;
pub use context::*;
pub use http::*;
pub use json::*;
pub use model::*;
pub use tool::*;

/// A type alias for a boxed error that is thread-safe and sendable across threads.
/// This is commonly used as a return type for functions that can return various error types.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A type alias for a boxed future that is thread-safe and sendable across threads.
pub type BoxPinFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Converts a path to lowercase path.
pub fn path_lowercase(path: &Path) -> Path {
    Path::from(path.as_ref().to_ascii_lowercase())
}

/// Validates a path part to ensure it doesn't contain the path delimiter
/// agent name and user name should be validated.
pub fn validate_path_part(part: &str) -> Result<(), BoxError> {
    if part.is_empty() || part.contains(DELIMITER) || Path::from(part).as_ref() != part {
        return Err(format!("invalid path part: {}", part).into());
    }

    Ok(())
}

/// Validates a function name to ensure it doesn't contain invalid characters
///
/// # Rules
/// - Must not be empty
/// - Must not exceed 64 characters
/// - Must start with a lowercase letter
/// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_)
pub fn validate_function_name(name: &str) -> Result<(), BoxError> {
    if name.is_empty() {
        return Err("empty string".into());
    }

    if name.len() > 64 {
        return Err("string length exceeds the limit 64".into());
    }

    let mut iter = name.chars();
    if !matches!(iter.next(), Some('a'..='z')) {
        return Err("name must start with a lowercase letter".into());
    }

    for c in iter {
        if !matches!(c, 'a'..='z' | '0'..='9' | '_' ) {
            return Err(format!("invalid character: {}", c).into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_lowercase() {
        let a = Path::from("a/Foo");
        assert_eq!(path_lowercase(&a).as_ref(), "a/foo");
    }

    #[test]
    fn test_validate_path_part() {
        assert!(validate_path_part("foo").is_ok());
        assert!(validate_path_part("fOO").is_ok());
        assert!(validate_path_part("").is_err());
        assert!(validate_path_part("foo/").is_err());
        assert!(validate_path_part("/foo").is_err());
        assert!(validate_path_part("foo/bar").is_err());
        assert!(validate_path_part("foo/bar/").is_err());
    }
}
