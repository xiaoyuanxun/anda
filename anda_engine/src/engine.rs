use anda_core::{http_rpc, AgentOutput, AgentSet, BoxError, FunctionDefinition, ToolSet};
use candid::Principal;
use ic_cose_types::validate_str;
use ic_tee_agent::http::HEADER_IC_TEE_SESSION;
use object_store::memory::InMemory;
use std::{sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;

use crate::{
    context::{AgentCtx, BaseCtx},
    model::Model,
    store::{Store, VectorStore},
    APP_USER_AGENT,
};

static TEE_LOCAL_SERVER: &str = "http://127.0.0.1:8080";

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

    /// Cancel all tasks in engine.
    pub fn cancel(&self) {
        self.ctx.base.cancellation_token.cancel()
    }

    pub async fn agent_run(
        &self,
        user: String,
        caller: Option<Principal>,
        prompt: String,
        attachment: Option<Vec<u8>>,
        agent_name: Option<String>,
    ) -> Result<AgentOutput, BoxError> {
        let name = agent_name.unwrap_or(self.default_agent.clone());
        if !self.ctx.agents.contains(&name) {
            return Err(format!("agent {} not found", name).into());
        }

        let ctx = self.ctx.child_with(&name, user, caller)?;
        self.ctx.agents.run(&name, ctx, prompt, attachment).await
    }

    pub async fn tool_call(
        &self,
        user: String,
        caller: Option<Principal>,
        name: String,
        args: String,
    ) -> Result<String, BoxError> {
        if !self.ctx.tools.contains(&name) {
            return Err(format!("tool {} not found", name).into());
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
    vector_store: VectorStore,
    tee_host: String,
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
            vector_store: VectorStore::not_implemented(),
            tee_host: TEE_LOCAL_SERVER.to_string(),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn with_tee_host(mut self, tee_host: String) -> Self {
        self.tee_host = tee_host;
        self
    }

    pub fn with_cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = cancellation_token;
        self
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

    pub async fn build(self, default_agent: String) -> Result<Engine, BoxError> {
        if !self.agents.contains(&default_agent) {
            return Err(format!("default agent {} not found", default_agent).into());
        }

        let name = self.name;
        // register engine session
        // 注意，engine name 必须在 tee host 服务的 app 白名单中，且只能注册一次，重复注册会失败。
        let session: String = http_rpc(
            &reqwest::Client::new(),
            &format!("{}/identity", self.tee_host),
            "register_session",
            &(&name.to_ascii_lowercase(),),
        )
        .await?;

        let local_http = reqwest::Client::builder()
            .http2_keep_alive_interval(Some(Duration::from_secs(25)))
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .http2_keep_alive_while_idle(true)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(20))
            .user_agent(APP_USER_AGENT)
            .default_headers({
                let mut headers = http::header::HeaderMap::new();
                headers.insert(&HEADER_IC_TEE_SESSION, session.parse().unwrap());
                headers
            })
            .build()?;

        let ctx = BaseCtx::new(
            &self.tee_host,
            self.cancellation_token.clone(),
            local_http.clone(),
            self.store,
        );

        let ctx = AgentCtx::new(
            ctx,
            self.model,
            self.vector_store,
            Arc::new(self.tools),
            Arc::new(self.agents),
        );

        Ok(Engine {
            ctx,
            name,
            default_agent,
        })
    }
}
