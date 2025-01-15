//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.

use anda_core::{
    Agent, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, FunctionDefinition, Tool,
};
use serde_json::{json, Value};
use std::marker::PhantomData;

pub use schemars::{schema_for, JsonSchema};
pub use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::context::{AgentCtx, BaseCtx};

pub struct SubmitTool<T: JsonSchema + DeserializeOwned + Send + Sync> {
    name: String,
    schema: Value,

    _t: PhantomData<T>,
}

impl<T> Default for SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    pub fn new() -> SubmitTool<T> {
        let mut schema = schema_for!(T);
        schema.meta_schema = None; // Remove the $schema field
        let name = schema
            .schema
            .metadata
            .as_ref()
            .and_then(|v| v.title.clone())
            .unwrap_or("tool".to_string())
            .to_ascii_lowercase();
        SubmitTool {
            name,
            schema: json!(schema),
            _t: PhantomData,
        }
    }
}

impl<T> Tool<BaseCtx> for SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    type Args = T;
    type Output = T;

    fn name(&self) -> String {
        format!("submit_{}", self.name)
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: "Submit the structured data you extracted from the provided text."
                .to_string(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(&self, _ctx: BaseCtx, data: Self::Args) -> Result<Self::Output, BoxError> {
        Ok(data)
    }
}

/// Extractor for structured data from text
pub struct Extractor<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync>(pub SubmitTool<T>);

impl<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync> Default for Extractor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync> Extractor<T> {
    pub fn new() -> Self {
        Self(SubmitTool::new())
    }

    pub async fn extract(
        &self,
        ctx: &AgentCtx,
        prompt: String,
    ) -> Result<(T, AgentOutput), BoxError> {
        let tool = self.0.name();
        let req = CompletionRequest {
            system: Some(format!("\
                You are an AI assistant whose purpose is to\
                extract structured data from the provided text.\n\
                You will have access to a `{tool}` function that defines the structure of the data to extract from the provided text.\n\
                Use the `{tool}` function to submit the structured data.\n\
                Be sure to fill out every field and ALWAYS CALL THE `{tool}` function, event with default values!!!.")),
            prompt,
            tools: vec![self.0.definition()],
            tool_choice_required: true,
            ..Default::default()
        };

        let mut res = ctx.completion(req).await?;
        if let Some(tool_calls) = &mut res.tool_calls {
            if let Some(tool) = tool_calls.iter_mut().next() {
                let result = self.0.call_raw(ctx.base.clone(), tool.args.clone()).await?;
                let t: T = serde_json::from_str(&result)?;
                tool.result = Some(result);
                return Ok((t, res));
            }
        }

        Err(format!("extract with {tool} failed, no tool_calls").into())
    }
}

impl<T> Agent<AgentCtx> for Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    fn name(&self) -> String {
        format!("{}_extractor", self.0.name)
    }

    fn description(&self) -> String {
        "Extract structured data from text using LLMs.".to_string()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _attachment: Option<Vec<u8>>,
    ) -> Result<AgentOutput, BoxError> {
        let (_, res) = self.extract(&ctx, prompt).await?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use anda_core::AgentContext;

    use super::*;
    use crate::{engine::EngineBuilder, model::Model};

    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }

    #[test]
    fn test_definition() {
        let tool = SubmitTool::<TestStruct>::new();
        let definition = tool.definition();
        assert_eq!(definition.name, "submit_teststruct");
        let s = serde_json::to_string(&definition).unwrap();
        assert!(s.contains(r#""required":["name"]"#));
        assert!(!s.contains("$schema"));
        println!("{}", s);
        // {"name":"submit_teststruct","description":"Submit the structured data you extracted from the provided text.","parameters":{"properties":{"age":{"format":"uint8","minimum":0.0,"type":["integer","null"]},"name":{"type":"string"}},"required":["name"],"title":"TestStruct","type":"object"},"strict":true}

        let agent = Extractor::<TestStruct>::new();
        let definition = agent.definition();
        assert_eq!(definition.name, "teststruct_extractor");
        let s = serde_json::to_string(&definition).unwrap();
        assert!(s.contains(r#""parameters":{"type":"string"}"#));
        assert!(!s.contains("$schema"));
        println!("{}", s);
        // {"name":"teststruct_extractor","description":"Extract structured data from text using LLMs.","parameters":{"type":"string"}}
    }

    #[tokio::test]
    async fn test_with_ctx() {
        let tool = SubmitTool::<TestStruct>::new();
        let agent = Extractor::<TestStruct>::new();
        let tool_name = tool.name();
        let agent_name = agent.name();

        let ctx = EngineBuilder::new()
            .with_model(Model::mock_implemented())
            .register_tool(tool)
            .unwrap()
            .register_agent(agent)
            .unwrap()
            .mock_ctx();

        let res = ctx
            .tool_call(&tool_name, r#"{"name":"Anda","age": 1}"#.to_string())
            .await
            .unwrap();
        assert_eq!(res, r#"{"name":"Anda","age":1}"#);

        let res = ctx
            .tool_call(&tool_name, r#"{"name": "Anda"}"#.to_string())
            .await
            .unwrap();
        assert_eq!(res, r#"{"name":"Anda","age":null}"#);

        let res = ctx
            .tool_call(&tool_name, r#"{"name": 123}"#.to_string())
            .await;
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("invalid args"));

        let res = ctx
            .agent_run(&agent_name, r#"{"name": "Anda"}"#.to_string(), None)
            .await
            .unwrap();
        assert_eq!(
            res.tool_calls.unwrap()[0].result,
            Some(r#"{"name":"Anda","age":null}"#.to_string())
        );

        let res = ctx
            .agent_run(&agent_name, r#"{"name": 123}"#.to_string(), None)
            .await;
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("invalid args"));
    }
}
