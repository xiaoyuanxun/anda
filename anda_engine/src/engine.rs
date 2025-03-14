//! The Engine module provides the core functionality for managing and executing agents and tools.
//!
//! # Overview
//! The Engine is the central component that orchestrates agent execution, tool management,
//! and context handling. It provides:
//! - Agent management and execution
//! - Tool registration and invocation
//! - Context management with cancellation support
//! - Builder pattern for configuration
//!
//! # Key Components
//! - [`Engine`]: The main struct that provides execution capabilities
//! - [`EngineBuilder`]: Builder pattern for constructing Engine instances
//! - Context management through [`AgentCtx`] and [`BaseCtx`]
//!
//! # Usage
//! 1. Create an Engine using the builder pattern
//! 2. Register tools and agents
//! 3. Execute agents or call tools
//!
//! # Example
//! ```rust,ignore
//! let engine = Engine::builder()
//!     .with_name("MyEngine".to_string())
//!     .register_tool(my_tool)?
//!     .register_agent(my_agent)?
//!     .build("default_agent".to_string())?;
//!
//! let output = engine.agent_run(None, "Hello".to_string(), None, None, None).await?;
//! ```

use anda_core::{
    Agent, AgentInput, AgentOutput, AgentSet, BoxError, Function, Metadata,
    Path, Tool, ToolInput, ToolOutput, ToolSet, Value, validate_function_name,
};
use async_trait::async_trait;
use candid::Principal;
use object_store::memory::InMemory;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;

use crate::{
    context::{AgentCtx, BaseCtx, Web3Client, Web3SDK},
    model::Model,
    store::Store,
};

pub use crate::context::{Information, InformationJSON, RemoteEngineArgs, RemoteEngines};

pub static ROOT_PATH: &str = "_";

/// Engine is the core component that manages agents, tools, and execution context.
/// It provides methods to interact with agents, call tools, and manage execution.
#[derive(Clone)]
pub struct Engine {
    id: Principal,
    ctx: AgentCtx,
    name: String,
    description: String,
    default_agent: String,
    export_agents: BTreeSet<String>,
    export_tools: BTreeSet<String>,
    hooks: Arc<Hooks>,
}

/// Hook trait for customizing engine behavior.
/// Hooks can be used to intercept and modify agent and tool execution.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Called before an agent is executed.
    async fn on_agent_start(&self, _ctx: &AgentCtx, _agent: &str) -> Result<(), BoxError> {
        Ok(())
    }

    /// Called after an agent is executed.
    async fn on_agent_end(
        &self,
        _ctx: &AgentCtx,
        _agent: &str,
        output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        Ok(output)
    }

    /// Called before a tool is called.
    async fn on_tool_start(&self, _ctx: &BaseCtx, _tool: &str) -> Result<(), BoxError> {
        Ok(())
    }

    /// Called after a tool is called.
    async fn on_tool_end(
        &self,
        _ctx: &BaseCtx,
        _tool: &str,
        output: ToolOutput<Value>,
    ) -> Result<ToolOutput<Value>, BoxError> {
        Ok(output)
    }
}

/// Hooks struct for managing multiple hooks.
pub struct Hooks {
    hooks: Vec<Box<dyn Hook>>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self::new()
    }
}

impl Hooks {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Adds a new hook to the list of hooks.
    pub fn add(&mut self, hook: Box<dyn Hook>) {
        self.hooks.push(hook);
    }
}

#[async_trait]
impl Hook for Hooks {
    async fn on_agent_start(&self, ctx: &AgentCtx, agent: &str) -> Result<(), BoxError> {
        for hook in &self.hooks {
            hook.on_agent_start(ctx, agent).await?;
        }
        Ok(())
    }

    async fn on_agent_end(
        &self,
        ctx: &AgentCtx,
        agent: &str,
        mut output: AgentOutput,
    ) -> Result<AgentOutput, BoxError> {
        for hook in &self.hooks {
            output = hook.on_agent_end(ctx, agent, output).await?;
        }
        Ok(output)
    }

    async fn on_tool_start(&self, ctx: &BaseCtx, tool: &str) -> Result<(), BoxError> {
        for hook in &self.hooks {
            hook.on_tool_start(ctx, tool).await?;
        }
        Ok(())
    }

    async fn on_tool_end(
        &self,
        ctx: &BaseCtx,
        tool: &str,
        mut output: ToolOutput<Value>,
    ) -> Result<ToolOutput<Value>, BoxError> {
        for hook in &self.hooks {
            output = hook.on_tool_end(ctx, tool, output).await?;
        }
        Ok(output)
    }
}

impl Engine {
    /// Creates a new EngineBuilder instance for constructing an Engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// Returns the engine ID.
    pub fn id(&self) -> Principal {
        self.id
    }

    /// Returns the name of the engine.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Returns the description of the engine.
    pub fn description(&self) -> String {
        self.description.clone()
    }

    /// Returns the name of the default agent.
    pub fn default_agent(&self) -> String {
        self.default_agent.clone()
    }

    /// Cancels all tasks in the engine by triggering the cancellation token.
    pub fn cancel(&self) {
        self.ctx.base.cancellation_token.cancel()
    }

    /// Creates and returns a child cancellation token.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.ctx.base.cancellation_token.child_token()
    }

    /// Creates a new [`AgentCtx`] with the specified agent name, user, and caller.
    /// Returns an error if the agent is not found or if the user name is invalid.
    pub fn ctx_with(
        &self,
        caller: Principal,
        agent_name: &str,
        meta: Metadata,
    ) -> Result<AgentCtx, BoxError> {
        let name = agent_name.to_ascii_lowercase();
        if !self.export_agents.contains(&name) || !self.ctx.agents.contains(&name) {
            return Err(format!("agent {} not found", name).into());
        }

        self.ctx.child_with(caller, &name, meta)
    }

    /// Executes an agent with the specified parameters.
    /// If no agent name is provided, uses the default agent.
    /// Returns the agent's output or an error if the agent is not found.
    pub async fn agent_run(
        &self,
        caller: Principal,
        mut input: AgentInput,
    ) -> Result<AgentOutput, BoxError> {
        input.name = if input.name.is_empty() {
            self.default_agent.clone()
        } else {
            input.name.to_ascii_lowercase()
        };

        let agent = self
            .ctx
            .agents
            .get(&input.name)
            .ok_or_else(|| format!("agent {} not found", input.name))?;
        let ctx = self.ctx_with(caller, &input.name, input.meta.unwrap_or_default())?;

        self.hooks.on_agent_start(&ctx, &input.name).await?;
        let mut res = agent
            .run(ctx.clone(), input.prompt, input.resources)
            .await?;
        res.full_history = None; // clear full history
        self.hooks.on_agent_end(&ctx, &input.name, res).await
    }

    /// Calls a tool by name with the specified arguments.
    /// Returns tuple containing the result string and a boolean indicating if further processing is needed.
    pub async fn tool_call(
        &self,
        caller: Principal,
        input: ToolInput<Value>,
    ) -> Result<ToolOutput<Value>, BoxError> {
        if !self.export_tools.contains(&input.name) || !self.ctx.tools.contains(&input.name) {
            return Err(format!("tool {} not found", &input.name).into());
        }

        let tool = self
            .ctx
            .tools
            .get(&input.name)
            .ok_or_else(|| format!("tool {} not found", &input.name))?;

        let ctx = self
            .ctx
            .child_base_with(caller, &input.name, input.meta.unwrap_or_default())?;
        self.hooks.on_tool_start(&ctx, &input.name).await?;
        let args = serde_json::to_string(&input.args)?;
        let res = tool.call(ctx.clone(), args, input.resources).await?;
        self.hooks.on_tool_end(&ctx, &input.name, res).await
    }

    /// Returns function definitions for the specified agents.
    /// If no names are provided, returns definitions for all agents.
    pub fn agents(&self, names: Option<&[&str]>) -> Vec<Function> {
        self.ctx.agents.functions(names)
    }

    /// Returns function definitions for the specified tools.
    /// If no names are provided, returns definitions for all tools.
    pub fn tools(&self, names: Option<&[&str]>) -> Vec<Function> {
        self.ctx.tools.functions(names)
    }

    /// Returns information about the engine, including agent and tool definitions.
    pub fn information(&self) -> Information {
        Information {
            id: self.id,
            name: self.name.clone(),
            description: self.description.clone(),
            endpoint: "".to_string(),
            agents: self.agents(Some(
                self.export_agents
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )),
            tools: self.tools(Some(
                self.export_tools
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )),
        }
    }
}

/// Builder pattern implementation for constructing an Engine.
/// Allows for step-by-step configuration of the engine's components.
pub struct EngineBuilder {
    id: Principal,
    name: String,
    description: String,
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    remote: BTreeMap<String, RemoteEngineArgs>,
    model: Model,
    store: Store,
    web3: Arc<Web3SDK>,
    hooks: Arc<Hooks>,
    cancellation_token: CancellationToken,
    export_agents: BTreeSet<String>,
    export_tools: BTreeSet<String>,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    /// Creates a new EngineBuilder with default values.
    pub fn new() -> Self {
        let mstore = Arc::new(InMemory::new());
        EngineBuilder {
            id: Principal::anonymous(),
            name: "Anda".to_string(),
            description: "Anda Engine".to_string(),
            tools: ToolSet::new(),
            agents: AgentSet::new(),
            remote: BTreeMap::new(),
            model: Model::not_implemented(),
            store: Store::new(mstore),
            web3: Arc::new(Web3SDK::Web3(Web3Client::not_implemented())),
            hooks: Arc::new(Hooks { hooks: Vec::new() }),
            cancellation_token: CancellationToken::new(),
            export_agents: BTreeSet::new(),
            export_tools: BTreeSet::new(),
        }
    }

    /// Sets the engine ID, which comes from the TEE.
    pub fn with_id(mut self, id: Principal) -> Self {
        self.id = id;
        self
    }

    /// Sets the engine name.
    pub fn with_name(mut self, name: String) -> Result<Self, BoxError> {
        validate_function_name(&name.to_ascii_lowercase())?;
        self.name = name;
        Ok(self)
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Sets the cancellation token.
    pub fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = cancellation_token;
        self
    }

    /// Sets the TEE (Trusted Execution Environment) client.
    pub fn with_web3_client(mut self, web3: Arc<Web3SDK>) -> Self {
        self.web3 = web3;
        self
    }

    /// Sets the model to be used by the engine.
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    /// Sets the storage backend for the engine.
    pub fn with_store(mut self, store: Store) -> Self {
        self.store = store;
        self
    }

    /// Registers a single tool with the engine.
    /// Returns an error if the tool cannot be added.
    pub fn register_tool<T>(mut self, tool: T) -> Result<Self, BoxError>
    where
        T: Tool<BaseCtx> + Send + Sync + 'static,
    {
        self.tools.add(tool)?;
        Ok(self)
    }

    /// Registers multiple tools with the engine.
    /// Returns an error if any tool already exists.
    pub fn register_tools(mut self, tools: ToolSet<BaseCtx>) -> Result<Self, BoxError> {
        for (name, tool) in tools.set {
            if self.tools.set.contains_key(&name) {
                return Err(format!("tool {} already exists", name).into());
            }
            self.tools.set.insert(name, tool);
        }

        Ok(self)
    }

    /// Registers a single agent with the engine.
    /// Verifies that all required tools are registered before adding the agent.
    /// Returns an error if any dependency is missing or if the agent cannot be added.
    pub fn register_agent<T>(mut self, agent: T) -> Result<Self, BoxError>
    where
        T: Agent<AgentCtx> + Send + Sync + 'static,
    {
        for tool in agent.tool_dependencies() {
            if !self.tools.contains(&tool) {
                return Err(format!("dependent tool {} not found", tool).into());
            }
        }

        self.agents.add(agent)?;
        Ok(self)
    }

    /// Registers multiple agents with the engine.
    /// Verifies that all required tools are registered for each agent.
    /// Returns an error if any agent already exists or if any dependency is missing.
    pub fn register_agents(mut self, agents: AgentSet<AgentCtx>) -> Result<Self, BoxError> {
        for (name, agent) in agents.set {
            if self.agents.set.contains_key(&name) {
                return Err(format!("agent {} already exists", name).into());
            }

            for tool in agent.tool_dependencies() {
                if !self.tools.contains(&tool) {
                    return Err(format!("dependent tool {} not found", tool).into());
                }
            }
            self.agents.set.insert(name, agent);
        }

        Ok(self)
    }

    /// Registers a remote engine with given endpoint, optional agents, tools, and alias name.
    pub fn register_remote_engine(mut self, engine: RemoteEngineArgs) -> Result<Self, BoxError> {
        if self.remote.contains_key(&engine.endpoint) {
            return Err(format!("remote engine {} already exists", engine.endpoint).into());
        }
        if let Some(name) = &engine.name {
            validate_function_name(name).map_err(|err| format!("invalid name: {}", err))?;
        }

        self.remote.insert(engine.endpoint.clone(), engine);
        Ok(self)
    }

    /// Exports agents by name.
    pub fn export_agents(mut self, agents: Vec<String>) -> Self {
        for mut agent in agents {
            agent.make_ascii_lowercase();
            self.export_agents.insert(agent);
        }
        self
    }

    /// Exports tools by name.
    pub fn export_tools(mut self, tools: Vec<String>) -> Self {
        for tool in tools {
            self.export_tools.insert(tool);
        }
        self
    }

    /// Sets the hooks for the engine.
    pub fn with_hooks(mut self, hooks: Arc<Hooks>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Finalizes the builder and creates an Engine instance.
    /// Requires a default agent name to be specified.
    /// Returns an error if the default agent is not found.
    pub async fn build(mut self, default_agent: String) -> Result<Engine, BoxError> {
        let default_agent = default_agent.to_ascii_lowercase();
        if !self.agents.contains(&default_agent) {
            return Err(format!("default agent {} not found", default_agent).into());
        }

        self.export_agents.insert(default_agent.clone());

        let mut names: BTreeSet<Path> = self
            .tools
            .set
            .keys()
            .map(|p| Path::from(format!("T:{}", p)))
            .chain(
                self.agents
                    .set
                    .keys()
                    .map(|p| Path::from(format!("A:{}", p))),
            )
            .collect();
        names.insert(Path::from(ROOT_PATH));
        let ctx = BaseCtx::new(
            self.id,
            self.name.clone(),
            self.cancellation_token,
            names,
            self.web3,
            self.store,
        );

        let mut remote = RemoteEngines::new();
        for (_, engine) in self.remote {
            remote.register(ctx.clone(), engine).await?;
        }

        let tools = Arc::new(self.tools);
        let agents = Arc::new(self.agents);
        let ctx = AgentCtx::new(
            ctx,
            self.model,
            tools.clone(),
            agents.clone(),
            Arc::new(remote),
        );

        let meta = Metadata::default();
        for (name, tool) in &tools.set {
            let ct = ctx.child_base_with(self.id, name, meta.clone())?;
            tool.init(ct).await?;
        }

        for (name, agent) in &agents.set {
            let ct = ctx.child_with(self.id, name, meta.clone())?;
            agent.init(ct).await?;
        }

        Ok(Engine {
            id: self.id,
            ctx,
            name: self.name,
            description: self.description,
            default_agent,
            export_agents: self.export_agents,
            export_tools: self.export_tools,
            hooks: self.hooks,
        })
    }

    /// Creates a mock context for testing purposes.
    #[cfg(test)]
    pub fn mock_ctx(self) -> AgentCtx {
        let mut names: BTreeSet<Path> = self
            .tools
            .set
            .keys()
            .chain(self.agents.set.keys())
            .map(|s| Path::from(s.as_str()))
            .collect();
        names.insert(Path::from(ROOT_PATH));
        let ctx = BaseCtx::new(
            anda_core::ANONYMOUS,
            "Mocker".to_string(),
            self.cancellation_token,
            names,
            self.web3,
            self.store,
        );
        AgentCtx::new(
            ctx,
            self.model,
            Arc::new(self.tools),
            Arc::new(self.agents),
            Arc::new(RemoteEngines::new()),
        )
    }
}
