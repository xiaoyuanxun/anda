use object_store::path::DELIMITER;
use std::{future::Future, pin::Pin};

pub mod agent;
pub mod context;
pub mod http;
pub mod model;
pub mod tool;

pub use agent::*;
pub use context::*;
pub use http::*;
pub use model::*;
pub use tool::*;

/// A type alias for a boxed error that is thread-safe and sendable across threads.
/// This is commonly used as a return type for functions that can return various error types.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A type alias for a boxed future that is thread-safe and sendable across threads.
pub type BoxPinFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Joins two paths together
pub fn join_path(a: &Path, b: &Path) -> Path {
    Path::from(format!("{}/{}", a, b))
}

/// Validates a path part to ensure it doesn't contain the path delimiter
/// agent name and user name should be validated.
pub fn validate_path_part(part: &str) -> Result<(), BoxError> {
    if part.is_empty() || part.contains(DELIMITER) || Path::from(part).as_ref() != part {
        return Err(format!("invalid path part: {}", part).into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_path() {
        let a = Path::from("a/foo/");
        let b = Path::from("/b/bar");
        assert_eq!(a.as_ref(), "a/foo");
        assert_eq!(b.as_ref(), "b/bar");
        assert_eq!(join_path(&a, &b), Path::from("a/foo/b/bar"));
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
