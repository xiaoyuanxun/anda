
pub mod agent;
pub mod context;
pub mod tool;

/// A type alias for a boxed error that is thread-safe and sendable across threads.
/// This is commonly used as a return type for functions that can return various error types.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Represents an HTTP response with status code, headers, and optional body.
/// This struct is used to encapsulate HTTP response data in a structured way.
#[derive(Debug, Clone, Default)]
pub struct HttpResponse {
    pub status: http::StatusCode,
    pub headers: http::header::HeaderMap,
    pub body: Option<bytes::Bytes>,
}
