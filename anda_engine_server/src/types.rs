use anda_engine::engine::{Information, InformationJSON};
use candid::Principal;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformation {
    pub engines: Vec<Information>,
    pub default_engine: Principal,
    pub caller: Principal,
    pub start_time_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformationJSON {
    pub engines: Vec<InformationJSON>,
    pub default_engine: String,
    pub caller: String,
    pub start_time_ms: u64,
}

impl From<AppInformation> for AppInformationJSON {
    fn from(info: AppInformation) -> Self {
        AppInformationJSON {
            engines: info
                .engines
                .into_iter()
                .map(InformationJSON::from)
                .collect(),
            default_engine: info.default_engine.to_string(),
            caller: info.caller.to_string(),
            start_time_ms: info.start_time_ms,
        }
    }
}
