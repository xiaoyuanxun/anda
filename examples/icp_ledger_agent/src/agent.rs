use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CanisterCaller, CompletionFeatures,
    CompletionRequest, StateFeatures, ToolSet,
};
use anda_engine::context::{AgentCtx, BaseCtx};
use anda_icp::ledger::{BalanceOfTool, ICPLedgers, TransferTool};
use candid::Principal;
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone)]
pub struct ICPLedgerAgent {
    ledgers: Arc<ICPLedgers>,
    tools: Vec<&'static str>,
}

impl ICPLedgerAgent {
    pub const NAME: &'static str = "icp_ledger_agent";

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

    pub fn tools(&self) -> Result<ToolSet<BaseCtx>, BoxError> {
        let mut tools = ToolSet::new();
        tools.add(BalanceOfTool::new(self.ledgers.clone()))?;
        tools.add(TransferTool::new(self.ledgers.clone()))?;
        Ok(tools)
    }
}

impl Agent<AgentCtx> for ICPLedgerAgent {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Interacts with ICP blockchain ledgers".to_string()
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.tools.iter().map(|v| v.to_string()).collect()
    }

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
