use anda_core::{BoxError, FunctionDefinition, HttpFeatures, validate_function_name};
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
    /// The default agent for the engine.
    pub default_agent: String,
    /// Definitions for agents in the engine.
    pub agent_definitions: Vec<FunctionDefinition>,
    /// Definitions for tools in the engine.
    pub tool_definitions: Vec<FunctionDefinition>,
    /// The endpoint of the engine. It may be empty.
    pub endpoint: String,
}

/// Information about the engine in JSON format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InformationJSON {
    pub id: String,
    pub name: String,
    pub default_agent: String,
    pub agent_definitions: Vec<FunctionDefinition>,
    pub tool_definitions: Vec<FunctionDefinition>,
    pub endpoint: String,
}

impl From<Information> for InformationJSON {
    fn from(info: Information) -> Self {
        InformationJSON {
            id: info.id.to_text(),
            name: info.name,
            default_agent: info.default_agent,
            agent_definitions: info.agent_definitions,
            tool_definitions: info.tool_definitions,
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
            let agent_definitions: Vec<FunctionDefinition> = info
                .agent_definitions
                .into_iter()
                .filter(|d| args.agents.contains(&d.name))
                .collect();
            for agent in args.agents {
                if !agent_definitions.iter().any(|d| d.name == agent) {
                    return Err(format!("agent {:?} not found in engine {:?}", agent, name).into());
                }
            }

            info.agent_definitions = agent_definitions;
        }

        if !args.tools.is_empty() {
            let tool_definitions: Vec<FunctionDefinition> = info
                .tool_definitions
                .into_iter()
                .filter(|d| args.tools.is_empty() || args.tools.contains(&d.name))
                .collect();
            for tool in args.tools {
                if !tool_definitions.iter().any(|d| d.name == tool) {
                    return Err(format!("tool {:?} not found in engine {:?}", tool, name).into());
                }
            }
            info.tool_definitions = tool_definitions;
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
                        .tool_definitions
                        .iter()
                        .filter_map(|d| {
                            if let Some(names) = names {
                                if names.contains(&d.name.as_str()) {
                                    Some(d.clone().name_with_prefix(&prefix))
                                } else {
                                    None
                                }
                            } else {
                                Some(d.clone().name_with_prefix(&prefix))
                            }
                        })
                        .collect();
                }
            }
        }

        let mut definitions = Vec::with_capacity(
            self.engines
                .values()
                .map(|e| e.tool_definitions.len())
                .sum(),
        );
        for (prefix, engine) in self.engines.iter() {
            let prefix = format!("RT_{prefix}");
            definitions.extend(engine.tool_definitions.iter().filter_map(|d| {
                if let Some(names) = names {
                    if names.contains(&d.name.as_str()) {
                        Some(d.clone().name_with_prefix(&prefix))
                    } else {
                        None
                    }
                } else {
                    Some(d.clone().name_with_prefix(&prefix))
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
                        .agent_definitions
                        .iter()
                        .filter_map(|d| {
                            if let Some(names) = names {
                                if names.contains(&d.name.as_str()) {
                                    Some(d.clone().name_with_prefix(&prefix))
                                } else {
                                    None
                                }
                            } else {
                                Some(d.clone().name_with_prefix(&prefix))
                            }
                        })
                        .collect();
                }
            }
        }

        let mut definitions = Vec::with_capacity(
            self.engines
                .values()
                .map(|e| e.agent_definitions.len())
                .sum(),
        );
        for (prefix, engine) in self.engines.iter() {
            let prefix = format!("RA_{prefix}");
            definitions.extend(engine.agent_definitions.iter().filter_map(|d| {
                if let Some(names) = names {
                    if names.contains(&d.name.as_str()) {
                        Some(d.clone().name_with_prefix(&prefix))
                    } else {
                        None
                    }
                } else {
                    Some(d.clone().name_with_prefix(&prefix))
                }
            }));
        }

        definitions
    }
}
