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
    validate_path_part, Agent, AgentOutput, AgentSet, BoxError, FunctionDefinition, Tool, ToolSet,
};
use candid::Principal;
use object_store::memory::InMemory;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::{
    context::{AgentCtx, BaseCtx, Web3Client, Web3SDK},
    model::Model,
    store::Store,
};

pub static ROOT_PATH: &str = "_";

/// Engine is the core component that manages agents, tools, and execution context.
/// It provides methods to interact with agents, call tools, and manage execution.
#[derive(Clone)]
pub struct Engine {
    id: Principal,
    ctx: AgentCtx,
    name: String, // engine name
    default_agent: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Information {
    pub id: Principal,
    pub name: String,
    pub default_agent: String,
    pub agent_definitions: Vec<FunctionDefinition>,
    pub tool_definitions: Vec<FunctionDefinition>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InformationJSON {
    pub id: String,
    pub name: String,
    pub default_agent: String,
    pub agent_definitions: Vec<FunctionDefinition>,
    pub tool_definitions: Vec<FunctionDefinition>,
}

impl From<Information> for InformationJSON {
    fn from(info: Information) -> Self {
        InformationJSON {
            id: info.id.to_text(),
            name: info.name,
            default_agent: info.default_agent,
            agent_definitions: info.agent_definitions,
            tool_definitions: info.tool_definitions,
        }
    }
}

impl Engine {
    /// Creates a new EngineBuilder instance for constructing an Engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    pub fn id(&self) -> Principal {
        self.id
    }

    /// Returns the name of the engine.
    pub fn name(&self) -> String {
        self.name.clone()
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

    /// Creates a new [`AgentCtx`] with the specified agent, user, and caller.
    /// Returns an error if the agent is not found or if the user name is invalid.
    pub fn ctx_with<A>(
        &self,
        agent: &A,
        user: Option<String>,
        caller: Option<Principal>,
    ) -> Result<AgentCtx, BoxError>
    where
        A: Agent<AgentCtx>,
    {
        let name = agent.name().to_ascii_lowercase();
        if !self.ctx.agents.contains(&name) {
            return Err(format!("agent {} not found", name).into());
        }
        if let Some(user) = &user {
            validate_path_part(user)?;
        }

        self.ctx.child_with(&name, user, caller)
    }

    /// Executes an agent with the specified parameters.
    /// If no agent name is provided, uses the default agent.
    /// Returns the agent's output or an error if the agent is not found.
    pub async fn agent_run(
        &self,
        agent_name: Option<String>,
        prompt: String,
        attachment: Option<ByteBuf>,
        user: Option<String>,
        caller: Option<Principal>,
    ) -> Result<AgentOutput, BoxError> {
        let name = agent_name.unwrap_or(self.default_agent.clone());
        if !self.ctx.agents.contains(&name) {
            return Err(format!("agent {} not found", name).into());
        }

        if let Some(user) = &user {
            validate_path_part(user)?;
        }

        let ctx = self.ctx.child_with(&name, user, caller)?;
        let mut res = self
            .ctx
            .agents
            .run(&name, ctx, prompt, attachment.map(|v| v.into_vec()))
            .await?;
        res.full_history = None; // clear full history
        Ok(res)
    }

    /// Calls a tool by name with the specified arguments.
    /// Returns tuple containing the result string and a boolean indicating if further processing is needed.
    pub async fn tool_call(
        &self,
        name: String,
        args: String,
        user: Option<String>,
        caller: Option<Principal>,
    ) -> Result<(String, bool), BoxError> {
        if !self.ctx.tools.contains(&name) {
            return Err(format!("tool {} not found", name).into());
        }

        if let Some(user) = &user {
            validate_path_part(user)?;
        }

        let ctx = self.ctx.child_base_with(&name, user, caller)?;
        self.ctx.tools.call(&name, ctx, args).await
    }

    /// Returns function definitions for the specified agents.
    /// If no names are provided, returns definitions for all agents.
    pub fn agent_definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        self.ctx.agents.definitions(names)
    }

    /// Returns function definitions for the specified tools.
    /// If no names are provided, returns definitions for all tools.
    pub fn tool_definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        self.ctx.tools.definitions(names)
    }

    pub fn information(&self) -> Information {
        Information {
            id: self.id,
            name: self.name.clone(),
            default_agent: self.default_agent.clone(),
            agent_definitions: self.agent_definitions(None),
            tool_definitions: self.tool_definitions(None),
        }
    }
}

/// Builder pattern implementation for constructing an Engine.
/// Allows for step-by-step configuration of the engine's components.
pub struct EngineBuilder {
    id: Principal,
    name: String,
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    model: Model,
    store: Store,
    web3: Arc<Web3SDK>,
    cancellation_token: CancellationToken,
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
            tools: ToolSet::new(),
            agents: AgentSet::new(),
            model: Model::not_implemented(),
            store: Store::new(mstore),
            web3: Arc::new(Web3SDK::Web3(Web3Client::not_implemented())),
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Sets the engine ID, which comes from the TEE.
    pub fn with_id(mut self, id: Principal) -> Self {
        self.id = id;
        self
    }

    /// Sets the engine name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
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

    /// Finalizes the builder and creates an Engine instance.
    /// Requires a default agent name to be specified.
    /// Returns an error if the default agent is not found.
    pub fn build(self, default_agent: String) -> Result<Engine, BoxError> {
        if !self.agents.contains(&default_agent) {
            return Err(format!("default agent {} not found", default_agent).into());
        }

        let ctx = BaseCtx::new(self.id, self.cancellation_token, self.web3, self.store);
        let ctx = AgentCtx::new(ctx, self.model, Arc::new(self.tools), Arc::new(self.agents));

        Ok(Engine {
            id: self.id,
            ctx,
            name: self.name,
            default_agent,
        })
    }

    /// Creates a mock context for testing purposes.
    #[cfg(test)]
    pub fn mock_ctx(self) -> AgentCtx {
        let ctx = BaseCtx::new(
            Principal::anonymous(),
            self.cancellation_token,
            self.web3,
            self.store,
        );
        AgentCtx::new(ctx, self.model, Arc::new(self.tools), Arc::new(self.agents))
    }
}
