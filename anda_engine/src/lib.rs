pub mod context;
pub mod engine;
pub mod model;
pub mod plugin;
pub mod store;

pub static APP_USER_AGENT: &str = concat!(
    "Mozilla/5.0 anda.bot ",
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
);
