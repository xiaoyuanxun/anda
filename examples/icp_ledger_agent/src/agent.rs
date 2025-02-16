use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CanisterCaller, CompletionFeatures,
    CompletionRequest, StateFeatures, ToolSet,
};
use anda_engine::context::{AgentCtx, BaseCtx};
use anda_icp::ledger::{BalanceOfTool, ICPLedgers, TransferTool};
use candid::Principal;
use std::{collections::BTreeSet, sync::Arc};

/// An AI agent implementation for interacting with ICP blockchain ledgers.
/// This agent provides capabilities to check balances and transfer ICP tokens
/// using the provided tools.
#[derive(Clone)]
pub struct ICPLedgerAgent {
    ledgers: Arc<ICPLedgers>,
    tools: Vec<&'static str>,
}

impl ICPLedgerAgent {
    /// The name identifier for this agent.
    pub const NAME: &'static str = "icp_ledger_agent";

    /// Creates and initializes a new ICPLedgerAgent instance.
    ///
    /// # Arguments
    /// * `ctx` - The canister caller context for making ICP ledger calls.
    /// * `ledgers` - A list of ledger canister IDs to interact with.
    ///
    /// # Returns
    /// A new ICPLedgerAgent instance or an error if initialization fails.
    pub async fn load(ctx: &impl CanisterCaller, ledgers: &[&str]) -> Result<Self, BoxError> {
        let ledgers: BTreeSet<Principal> = ledgers
            .iter()
            .flat_map(|v| Principal::from_text(v).map_err(|_| format!("invalid token: {}", v)))
            .collect();

        let ledgers = ICPLedgers::load(ctx, ledgers, false).await?;
        let ledgers = Arc::new(ledgers);
        Ok(Self {
            ledgers,
            tools: vec![BalanceOfTool::NAME, TransferTool::NAME],
        })
    }

    /// Returns the set of tools available through this agent.
    ///
    /// # Returns
    /// A ToolSet containing the balance check and transfer tools
    /// or an error if tool initialization fails.
    pub fn tools(&self) -> Result<ToolSet<BaseCtx>, BoxError> {
        let mut tools = ToolSet::new();
        tools.add(BalanceOfTool::new(self.ledgers.clone()))?;
        tools.add(TransferTool::new(self.ledgers.clone()))?;
        Ok(tools)
    }
}

/// Implementation of the [`Agent`] trait for ICPLedgerAgent.
impl Agent<AgentCtx> for ICPLedgerAgent {
    /// Returns the agent's name identifier
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Returns a description of the agent's purpose and capabilities.
    fn description(&self) -> String {
        "Interacts with ICP blockchain ledgers".to_string()
    }

    /// Returns a list of tool names that this agent depends on
    fn tool_dependencies(&self) -> Vec<String> {
        self.tools.iter().map(|v| v.to_string()).collect()
    }

    /// Main execution method for the agent.
    ///
    /// # Arguments
    /// * `ctx` - The agent context containing execution environment.
    /// * `prompt` - The user's input prompt.
    /// * `_attachment` - Optional binary attachment (not used).
    ///
    /// # Returns
    /// AgentOutput containing the response or an error if execution fails.
    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _attachment: Option<Vec<u8>>,
    ) -> Result<AgentOutput, BoxError> {
        let caller = ctx.caller().ok_or("missing caller")?;
        let req = CompletionRequest {
            system: Some(
                "\
            You are an AI assistant designed to interact with the ICP blockchain ledger by given tools.\n\
            1. Please decline any requests that are not related to the ICP blockchain ledger.\n\
            2. For requests that are not supported by the tools available, kindly inform the user \
            of your current capabilities."
                    .to_string(),
            ),
            prompt,
            tools: ctx.tool_definitions(Some(&self.tools)),
            tool_choice_required: false,
            ..Default::default()
        }
        .context("user_address".to_string(), caller.to_string());
        let res = ctx.completion(req).await?;
        Ok(res)
    }
}
