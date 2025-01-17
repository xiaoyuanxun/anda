use anda_core::{
    validate_path_part, Agent, AgentOutput, AgentSet, BoxError, FunctionDefinition, Tool, ToolSet,
};
use candid::Principal;
use object_store::memory::InMemory;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::{
    context::{AgentCtx, BaseCtx, TEEClient},
    model::Model,
    store::Store,
};

static TEE_LOCAL_SERVER: &str = "http://127.0.0.1:8080";

pub static ROOT_PATH: &str = "_";

#[derive(Clone)]
pub struct Engine {
    ctx: AgentCtx,
    name: String, // engine name
    default_agent: String,
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn default_agent(&self) -> String {
        self.default_agent.clone()
    }

    /// Cancel all tasks in engine.
    pub fn cancel(&self) {
        self.ctx.base.cancellation_token.cancel()
    }

    /// Return a child cancellation token.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.ctx.base.cancellation_token.child_token()
    }

    /// Return the agent context with user and caller.
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

    pub async fn agent_run(
        &self,
        agent_name: Option<String>,
        prompt: String,
        attachment: Option<Vec<u8>>,
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
        self.ctx.agents.run(&name, ctx, prompt, attachment).await
    }

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

    pub fn agent_definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        self.ctx.agents.definitions(names)
    }

    pub fn tool_definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        self.ctx.tools.definitions(names)
    }
}

pub struct EngineBuilder {
    name: String,
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    model: Model,
    store: Store,
    tee_client: TEEClient,
    cancellation_token: CancellationToken,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    pub fn new() -> Self {
        let mstore = Arc::new(InMemory::new());
        EngineBuilder {
            name: "Anda".to_string(),
            tools: ToolSet::new(),
            agents: AgentSet::new(),
            model: Model::not_implemented(),
            store: Store::new(mstore),
            tee_client: TEEClient::new(TEE_LOCAL_SERVER, ""),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = cancellation_token;
        self
    }

    pub fn with_tee_client(mut self, tee_client: TEEClient) -> Self {
        self.tee_client = tee_client;
        self
    }

    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    pub fn with_store(mut self, store: Store) -> Self {
        self.store = store;
        self
    }

    pub fn register_tool<T>(mut self, tool: T) -> Result<Self, BoxError>
    where
        T: Tool<BaseCtx> + Send + Sync + 'static,
    {
        self.tools.add(tool)?;
        Ok(self)
    }

    pub fn register_tools(mut self, tools: ToolSet<BaseCtx>) -> Result<Self, BoxError> {
        for (name, tool) in tools.set {
            if self.tools.set.contains_key(&name) {
                return Err(format!("tool {} already exists", name).into());
            }
            self.tools.set.insert(name, tool);
        }

        Ok(self)
    }

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

    pub fn build(self, default_agent: String) -> Result<Engine, BoxError> {
        if !self.agents.contains(&default_agent) {
            return Err(format!("default agent {} not found", default_agent).into());
        }

        let ctx = BaseCtx::new(self.cancellation_token, self.tee_client, self.store);
        let ctx = AgentCtx::new(ctx, self.model, Arc::new(self.tools), Arc::new(self.agents));

        Ok(Engine {
            ctx,
            name: self.name,
            default_agent,
        })
    }

    #[cfg(test)]
    pub fn mock_ctx(self) -> AgentCtx {
        let ctx = BaseCtx::new(self.cancellation_token, self.tee_client, self.store);
        AgentCtx::new(ctx, self.model, Arc::new(self.tools), Arc::new(self.agents))
    }
}
