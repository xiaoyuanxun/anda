use anda_core::{AgentOutput, AgentSet, BoxError, ToolSet};
use candid::Principal;
use ic_cose_types::validate_str;
use object_store::memory::InMemory;
use serde_json::Value;
use std::sync::Arc;

use crate::{
    context::{AgentCtx, BaseCtx},
    database::{Store, VectorStore},
    model::Model,
};

static TEE_LOCAL_SERVER: &str = "http://127.0.0.1:8080";

pub struct Engine {
    ctx: AgentCtx,
    default_agent: String,
}

impl Engine {
    pub async fn agent_run(
        &self,
        user: String,
        caller: Option<Principal>,
        prompt: String,
        attachment: Option<Value>,
        agent_name: Option<String>,
    ) -> Result<AgentOutput, BoxError> {
        let name = agent_name.unwrap_or(self.default_agent.clone());
        if !self.ctx.agents.contains(&name) {
            return Err(format!("agent {} not found", name).into());
        }

        let ctx = self.ctx.child_with(&name, user, caller)?;
        self.ctx.agents.run(&name, &ctx, &prompt, attachment).await
    }

    pub async fn tool_call(
        &self,
        user: String,
        caller: Option<Principal>,
        name: String,
        args: Value,
    ) -> Result<Value, BoxError> {
        if !self.ctx.tools.contains(&name) {
            return Err(format!("tool {} not found", name).into());
        }

        let ctx = self.ctx.child_base_with(&name, user, caller)?;
        self.ctx.tools.call(&name, &ctx, &args).await
    }
}

pub struct EngineBuilder {
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    model: Model,
    store: Store,
    vector_store: VectorStore,
    tee_host: String,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    pub fn new() -> Self {
        let ms = Arc::new(InMemory::new());
        EngineBuilder {
            tools: ToolSet::new(),
            agents: AgentSet::new(),
            model: Model::not_implemented(),
            store: Store::new(ms.clone()),
            vector_store: VectorStore::new(ms),
            tee_host: TEE_LOCAL_SERVER.to_string(),
        }
    }

    pub fn with_store(mut self, store: Store) -> Self {
        self.store = store;
        self
    }

    pub fn with_vector_store(mut self, vector_store: VectorStore) -> Self {
        self.vector_store = vector_store;
        self
    }

    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    pub fn register_tools(&mut self, tools: ToolSet<BaseCtx>) -> Result<(), BoxError> {
        for (name, tool) in tools.set {
            validate_str(&name)?;
            if self.tools.contains(&name) {
                return Err(format!("tool {} already exists", name).into());
            }
            self.tools.set.insert(name, tool);
        }

        Ok(())
    }

    pub fn register_agents(&mut self, agents: AgentSet<AgentCtx>) -> Result<(), BoxError> {
        for (name, agent) in agents.set {
            validate_str(&name)?;
            if self.agents.contains(&name) {
                return Err(format!("agent {} already exists", name).into());
            }
            self.agents.set.insert(name, agent);
        }

        Ok(())
    }

    pub fn build(self, default_agent: String) -> Engine {
        let http = reqwest::Client::new();
        let ctx = BaseCtx::new(&self.tee_host, http, self.store);
        let ctx = AgentCtx::new(
            ctx,
            self.model,
            self.vector_store,
            Arc::new(self.tools),
            Arc::new(self.agents),
        );

        if !ctx.agents.contains(&default_agent) {
            panic!("default agent {} not found", default_agent);
        }

        Engine { ctx, default_agent }
    }
}
