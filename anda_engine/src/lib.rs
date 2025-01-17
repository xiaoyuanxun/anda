use rand::Rng;

pub mod context;
pub mod engine;
pub mod extension;
pub mod model;
pub mod store;

/// Gets current unix timestamp in milliseconds
pub use structured_logger::unix_ms;

/// Generates N random bytes
pub use ic_cose::rand_bytes;

pub static APP_USER_AGENT: &str = concat!(
    "Mozilla/5.0 anda.bot ",
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
);

/// Generates a random number within the given range
pub fn rand_number<T, R>(range: R) -> T
where
    T: rand::distributions::uniform::SampleUniform,
    R: rand::distributions::uniform::SampleRange<T>,
{
    let mut rng = rand::thread_rng();
    rng.gen_range(range)
}
