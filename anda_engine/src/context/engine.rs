use anda_core::{
    Agent, AgentContext, AgentInput, AgentOutput, BaseContext, BoxError, Function,
    FunctionDefinition, HttpFeatures, Resource, Tool, ToolInput, ToolOutput, Value,
    select_resources, validate_function_name,
};
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::context::{AgentCtx, BaseCtx};

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

    /// Retrieves a remote engine ID by endpoint.
    pub fn get_id_by_endpoint(&self, endpoint: &str) -> Option<Principal> {
        for (_, engine) in self.engines.iter() {
            if engine.endpoint == endpoint {
                return Some(engine.id);
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

    /// Extracts resources from the provided list based on the tool's supported tags.
    pub fn select_tool_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> Option<Vec<Resource>> {
        if name.strip_prefix("RT_").is_some() {
            for (_, engine) in self.engines.iter() {
                for tool in engine.tools.iter() {
                    if tool.definition.name == name {
                        let tags: &[&str] = &tool
                            .supported_resource_tags
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<&str>>();
                        return select_resources(resources, tags);
                    }
                }
            }
        }
        None
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

    /// Extracts resources from the provided list based on the agent's supported tags.
    pub fn select_agent_resources(
        &self,
        name: &str,
        resources: &mut Vec<Resource>,
    ) -> Option<Vec<Resource>> {
        for (_, engine) in self.engines.iter() {
            for agent in engine.agents.iter() {
                if agent.definition.name == name {
                    let tags: &[&str] = &agent
                        .supported_resource_tags
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<&str>>();
                    return select_resources(resources, tags);
                }
            }
        }
        None
    }
}

/// Wraps a remote tool as a local tool.
#[derive(Debug, Clone)]
pub struct RemoteTool {
    engine: Principal,
    endpoint: String,
    function: Function,
    name: String,
}

impl RemoteTool {
    pub fn new(
        engine: Principal,
        endpoint: String,
        function: Function,
        name: Option<String>,
    ) -> Result<Self, BoxError> {
        let name = if let Some(name) = name {
            validate_function_name(&name)?;
            name
        } else {
            function.definition.name.clone()
        };

        Ok(Self {
            engine,
            endpoint,
            function,
            name,
        })
    }
}

impl Tool<BaseCtx> for RemoteTool {
    type Args = Value;
    type Output = Value;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.function.definition.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        let mut definition = self.function.definition.clone();
        definition.name = self.name.clone();
        definition
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.function.supported_resource_tags.clone()
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        ctx.remote_tool_call(
            &self.endpoint,
            ToolInput {
                name: self.function.definition.name.clone(),
                args,
                resources,
                meta: Some(ctx.self_meta(self.engine)),
            },
        )
        .await
    }
}

/// Wraps a remote agent as a local agent.
#[derive(Debug, Clone)]
pub struct RemoteAgent {
    engine: Principal,
    endpoint: String,
    function: Function,
    name: String,
}

impl RemoteAgent {
    pub fn new(
        engine: Principal,
        endpoint: String,
        function: Function,
        name: Option<String>,
    ) -> Result<Self, BoxError> {
        let name = if let Some(name) = name {
            validate_function_name(&name.to_ascii_lowercase())?;
            name
        } else {
            function.definition.name.clone()
        };

        Ok(Self {
            engine,
            endpoint,
            function,
            name,
        })
    }
}

impl Agent<AgentCtx> for RemoteAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.function.definition.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        let mut definition = self.function.definition.clone();
        definition.name = self.name.to_ascii_lowercase();
        definition
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.function.supported_resource_tags.clone()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Option<Vec<Resource>>,
    ) -> Result<AgentOutput, BoxError> {
        ctx.remote_agent_run(
            &self.endpoint,
            AgentInput {
                name: self.function.definition.name.clone(),
                prompt,
                resources,
                meta: Some(ctx.base.self_meta(self.engine)),
            },
        )
        .await
    }
}
