use anda_core::BoxError;
use config::{Config, File, FileFormat};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Conf {
    pub id_secret: String,
    pub root_secret: String,
    pub object_store: String,
    pub object_store_config: Option<BTreeMap<String, String>>,
}

impl Conf {
    pub fn from_file(file_name: &str) -> Result<Self, BoxError> {
        let builder = Config::builder().add_source(File::new(file_name, FileFormat::Toml));
        let cfg = builder.build()?.try_deserialize::<Conf>()?;
        Ok(cfg)
    }

    pub fn from_toml(content: &str) -> Result<Self, BoxError> {
        let cfg: Self = toml::from_str(content)?;
        Ok(cfg)
    }
}
