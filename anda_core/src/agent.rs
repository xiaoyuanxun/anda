use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;

use crate::context::BaseContext;
use crate::BoxError;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentDefinition {
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentOutput {
    pub message: String,
    pub failed_reason: Option<String>,
    pub tool_call: Option<(String, Value)>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    /// "system", "user", or "assistant"
    pub role: String,
    pub content: String,
}

#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    pub text: String,
    /// The embedding vector
    pub vec: Vec<f32>,
}

pub trait AgentContext: BaseContext {
    fn completion(
        &self,
        prompt: &str,
        json_output: bool,
        chat_history: &[Message],
    ) -> impl Future<Output = Result<AgentOutput, Self::Error>> + Send;

    fn tool_call(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> impl Future<Output = Result<Value, Self::Error>> + Send;

    fn remote_tool_call(
        &self,
        endpoint: &str,
        tool_name: &str,
        args: &Value,
    ) -> impl Future<Output = Result<Value, Self::Error>> + Send;

    fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::Error>> + Send;

    fn embed_query(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, Self::Error>> + Send;

    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    fn top_n<T>(
        &self,
        query: &str,
        n: usize,
    ) -> impl Future<Output = Result<Vec<(String, T)>, Self::Error>> + Send
    where
        T: DeserializeOwned;

    /// Same as `top_n` but returns the document ids only.
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<String>, Self::Error>> + Send;

    fn agent_run(
        &self,
        agent_name: &str,
        prompt: &str,
        attachment: Option<Value>,
    ) -> impl Future<Output = Result<AgentOutput, Self::Error>> + Send;

    fn remote_agent_run(
        &self,
        endpoint: &str,
        agent_name: &str,
        prompt: &str,
        attachment: Option<Value>,
    ) -> impl Future<Output = Result<AgentOutput, Self::Error>> + Send;
}

pub trait Agent<C>: Send + Sync
where
    C: AgentContext + Send + Sync + 'static,
{
    /// The name of the agent. This name should be unique in the engine.
    const NAME: &'static str;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// A method returning the agent definition. The user prompt can be used to
    /// tailor the definition to the specific use case.
    fn definition(&self, prompt: Option<&str>) -> AgentDefinition;

    fn run(
        &self,
        ctx: &C,
        prompt: &str,
        attachment: Option<&Value>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send + Sync;
}
