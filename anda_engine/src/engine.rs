use anda_core::{AgentOutput, AgentSet, BoxError, ToolSet};
use candid::Principal;
use ic_cose_types::validate_str;
use object_store::ObjectStore;
use serde_json::Value;
use std::sync::Arc;

use crate::context::{AgentCtx, BaseCtx};

pub struct Engine {
    ctx: AgentCtx,
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    store: Arc<dyn ObjectStore>,
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
        if !self.agents.contains(&name) {
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
        if !self.tools.contains(&name) {
            return Err(format!("tool {} not found", name).into());
        }

        let ctx = self.ctx.child_base_with(&name, user, caller)?;
        self.ctx.tools.call(&name, &ctx, &args).await
    }
}

pub struct EngineBuilder {
    tools: ToolSet<BaseCtx>,
    agents: AgentSet<AgentCtx>,
    store: Option<Box<dyn ObjectStore>>,
    default_agent: String,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    pub fn new() -> Self {
        EngineBuilder {
            tools: ToolSet::new(),
            agents: AgentSet::new(),
            store: None,
            default_agent: "default".to_string(),
        }
    }

    pub fn with_default_agent(mut self, name: String) -> Self {
        self.default_agent = name;
        self
    }

    pub fn with_store(mut self, store: Box<dyn ObjectStore>) -> Self {
        self.store = Some(store);
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

    // pub fn build(self) -> Engine {
    //     Engine {
    //         ctx: AgentCtx::new(),
    //         tools: self.tools,
    //         agents: self.agents,
    //         store: self.store,
    //         default_agent: self.default_agent,
    //     }
    // }
}
