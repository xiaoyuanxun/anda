use anda_core::{BoxError, Function, FunctionDefinition, HttpFeatures, validate_function_name};
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Information about the engine, including agent and tool definitions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Information {
    /// The principal ID of the engine.
    pub id: Principal,
    /// The name of the engine.
    pub name: String,
    /// Description of the engine.
    pub description: String,
    /// Definitions for agents in the engine.
    pub agents: Vec<Function>,
    /// Definitions for tools in the engine.
    pub tools: Vec<Function>,
    /// The endpoint of the engine. It can be empty if the engine is local.
    pub endpoint: String,
}

/// Information about the engine in JSON format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InformationJSON {
    pub id: String,
    pub name: String,
    pub description: String,
    pub agents: Vec<Function>,
    pub tools: Vec<Function>,
    pub endpoint: String,
}

impl From<Information> for InformationJSON {
    fn from(info: Information) -> Self {
        InformationJSON {
            id: info.id.to_text(),
            name: info.name,
            description: info.description,
            agents: info.agents,
            tools: info.tools,
            endpoint: info.endpoint,
        }
    }
}

/// Collection of remote engines.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteEngines {
    pub engines: BTreeMap<String, Information>,
}

/// Arguments for registering a remote engine.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteEngineArgs {
    /// The endpoint of the remote engine.
    pub endpoint: String,
    /// List of agents to include in the engine. If empty, all agents are included.
    pub agents: Vec<String>,
    /// List of tools to include in the engine. If empty, all tools are included.
    pub tools: Vec<String>,
    /// Optional name for the engine. If not provided, the engine name is used.
    pub name: Option<String>,
}

impl Default for RemoteEngines {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoteEngines {
    pub fn new() -> Self {
        Self {
            engines: BTreeMap::new(),
        }
    }

    /// Registers a remote engine with the given arguments.
    pub async fn register(
        &mut self,
        ctx: impl HttpFeatures,
        args: RemoteEngineArgs,
    ) -> Result<(), BoxError> {
        let mut info: Information = ctx
            .https_signed_rpc(&args.endpoint, "information", &(true,))
            .await?;
        let name = args.name.unwrap_or_else(|| info.name.to_ascii_lowercase());
        validate_function_name(&name)
            .map_err(|err| format!("invalid engine name {:?}: {}", &name, err))?;

        if !args.agents.is_empty() {
            let agents: Vec<Function> = info
                .agents
                .into_iter()
                .filter(|d| args.agents.contains(&d.definition.name))
                .collect();
            for agent in args.agents {
                if !agents.iter().any(|d| d.definition.name == agent) {
                    return Err(format!("agent {:?} not found in engine {:?}", agent, name).into());
                }
            }

            info.agents = agents;
        }

        if !args.tools.is_empty() {
            let tools: Vec<Function> = info
                .tools
                .into_iter()
                .filter(|d| args.tools.is_empty() || args.tools.contains(&d.definition.name))
                .collect();
            for tool in args.tools {
                if !tools.iter().any(|d| d.definition.name == tool) {
                    return Err(format!("tool {:?} not found in engine {:?}", tool, name).into());
                }
            }
            info.tools = tools;
        }

        info.endpoint = args.endpoint;
        self.engines.insert(name, info);
        Ok(())
    }

    /// Retrieves a remote tool endpoint and name from a prefixed name.
    pub fn get_tool_endpoint(&self, prefixed_name: &str) -> Option<(String, String)> {
        if let Some(name) = prefixed_name.strip_prefix("RT_") {
            for (prefix, engine) in self.engines.iter() {
                if let Some(tool_name) = name.strip_prefix(prefix) {
                    return Some((engine.endpoint.clone(), tool_name.to_string()));
                }
            }
        }
        None
    }

    /// Retrieves a remote agent endpoint and name from a prefixed name.
    pub fn get_agent_endpoint(&self, prefixed_name: &str) -> Option<(String, String)> {
        if let Some(name) = prefixed_name.strip_prefix("RA_") {
            for (prefix, engine) in self.engines.iter() {
                if let Some(agent_name) = name.strip_prefix(prefix) {
                    return Some((engine.endpoint.clone(), agent_name.to_string()));
                }
            }
        }
        None
    }

    /// Retrieves definitions for available tools in the remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint
    /// * `names` - Optional filter for specific tool names
    ///
    /// # Returns
    /// Vector of function definitions for the requested tools
    pub fn tool_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[&str]>,
    ) -> Vec<FunctionDefinition> {
        if let Some(endpoint) = endpoint {
            for (prefix, engine) in self.engines.iter() {
                if endpoint == engine.endpoint {
                    let prefix = format!("RT_{prefix}");
                    return engine
                        .tools
                        .iter()
                        .filter_map(|d| {
                            if let Some(names) = names {
                                if names.contains(&d.definition.name.as_str()) {
                                    Some(d.definition.clone().name_with_prefix(&prefix))
                                } else {
                                    None
                                }
                            } else {
                                Some(d.definition.clone().name_with_prefix(&prefix))
                            }
                        })
                        .collect();
                }
            }
        }

        let mut definitions =
            Vec::with_capacity(self.engines.values().map(|e| e.tools.len()).sum());

        for (prefix, engine) in self.engines.iter() {
            let prefix = format!("RT_{prefix}");
            definitions.extend(engine.tools.iter().filter_map(|d| {
                if let Some(names) = names {
                    if names.contains(&d.definition.name.as_str()) {
                        Some(d.definition.clone().name_with_prefix(&prefix))
                    } else {
                        None
                    }
                } else {
                    Some(d.definition.clone().name_with_prefix(&prefix))
                }
            }));
        }

        definitions
    }

    /// Retrieves definitions for available agents in the remote engines.
    ///
    /// # Arguments
    /// * `endpoint` - Optional filter for specific remote engine endpoint
    /// * `names` - Optional filter for specific agent names
    ///
    /// # Returns
    /// Vector of function definitions for the requested agents
    pub fn agent_definitions(
        &self,
        endpoint: Option<&str>,
        names: Option<&[&str]>,
    ) -> Vec<FunctionDefinition> {
        if let Some(endpoint) = endpoint {
            for (prefix, engine) in self.engines.iter() {
                if endpoint == engine.endpoint {
                    let prefix = format!("RA_{prefix}");
                    return engine
                        .agents
                        .iter()
                        .filter_map(|d| {
                            if let Some(names) = names {
                                if names.contains(&d.definition.name.as_str()) {
                                    Some(d.definition.clone().name_with_prefix(&prefix))
                                } else {
                                    None
                                }
                            } else {
                                Some(d.definition.clone().name_with_prefix(&prefix))
                            }
                        })
                        .collect();
                }
            }
        }

        let mut definitions =
            Vec::with_capacity(self.engines.values().map(|e| e.agents.len()).sum());
        for (prefix, engine) in self.engines.iter() {
            let prefix = format!("RA_{prefix}");
            definitions.extend(engine.agents.iter().filter_map(|d| {
                if let Some(names) = names {
                    if names.contains(&d.definition.name.as_str()) {
                        Some(d.definition.clone().name_with_prefix(&prefix))
                    } else {
                        None
                    }
                } else {
                    Some(d.definition.clone().name_with_prefix(&prefix))
                }
            }));
        }

        definitions
    }
}
