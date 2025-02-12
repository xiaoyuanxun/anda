use anda_core::BoxError;
use config::{Config, File, FileFormat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Icp {
    pub token_ledgers: Vec<String>,
}

/// Configuration for the LLM should be encrypted and stored in the ICP COSE canister.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Llm {
    #[serde(default)]
    pub deepseek_api_key: String,
    #[serde(default)]
    pub deepseek_endpoint: String,
    #[serde(default)]
    pub deepseek_model: String,
    #[serde(default)]
    pub cohere_api_key: String,
    #[serde(default)]
    pub cohere_embedding_model: String,
    #[serde(default)]
    pub openai_api_key: String,
    #[serde(default)]
    pub openai_endpoint: String,
    #[serde(default)]
    pub openai_embedding_model: String,
    #[serde(default)]
    pub openai_completion_model: String,
}

/// Configuration for the X should be encrypted and stored in the ICP COSE canister.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct X {
    pub username: String,
    pub password: String,
    pub email: Option<String>,
    pub two_factor_auth: Option<String>,
    pub cookie_string: Option<String>,
}

/// Configuration for the Google search should be encrypted and stored in the ICP COSE canister.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Google {
    pub api_key: String,
    pub search_engine_id: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Conf {
    pub llm: Llm,
    pub icp: Icp,
    pub x: X,
    pub google: Google,
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
