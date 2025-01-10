use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, future::Future, marker::PhantomData, pin::Pin};

use crate::{context::BaseContext, BoxError};

/// Defines the metadata and schema for a tool
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Core trait for implementing tools that can be used by the AI Agent system
pub trait Tool<C>: Send + Sync
where
    C: BaseContext + Send + Sync + 'static,
{
    /// The name of the tool. This name should be unique in the engine.
    const NAME: &'static str;

    /// Returns the tool's name
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Provides the tool's definition including its parameters schema. The user prompt can be used to
    /// tailor the definition to the specific use case.
    fn definition(&self, prompt: Option<&str>) -> ToolDefinition;

    /// Executes the tool with given context and arguments
    /// Both the arguments and return value are a JSON since these values are meant to
    /// be the input and output of LLM models (respectively)
    fn call(
        &self,
        ctx: &C,
        args: &Value,
    ) -> impl Future<Output = Result<Value, BoxError>> + Send + Sync;
}

/// Dynamic dispatch version of the Tool trait
pub trait ToolDyn<C>: Send + Sync
where
    C: BaseContext + Send + Sync + 'static,
{
    fn name(&self) -> String;

    fn definition(&self, prompt: Option<&str>) -> ToolDefinition;

    fn call<'a>(
        &'a self,
        ctx: &'a C,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, BoxError>> + Send + 'a>>;
}

/// Wrapper to convert static Tool implementation to dynamic dispatch
struct ToolWrapper<T, C>(T, PhantomData<fn() -> C>)
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static;

impl<T, C> ToolDyn<C> for ToolWrapper<T, C>
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static,
{
    fn name(&self) -> String {
        self.0.name()
    }

    fn definition(&self, prompt: Option<&str>) -> ToolDefinition {
        self.0.definition(prompt)
    }

    fn call<'a>(
        &'a self,
        ctx: &'a C,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, BoxError>> + Send + 'a>> {
        Box::pin(self.0.call(ctx, args))
    }
}

/// Possible errors when working with tools
#[derive(Debug, thiserror::Error)]
pub enum ToolSetError {
    /// Error returned by the tool
    #[error("tool call error: {0}")]
    ToolCallError(#[from] BoxError),

    #[error("tool not found: {0}")]
    ToolNotFoundError(String),
}

/// Collection of tools that can be used by the AI Agent
#[derive(Default)]
pub struct ToolSet<C: BaseContext + 'static> {
    pub(crate) tools: BTreeMap<String, Box<dyn ToolDyn<C>>>,
}

impl<C> ToolSet<C>
where
    C: BaseContext + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn contains(&self, toolname: &str) -> bool {
        self.tools.contains_key(toolname)
    }

    pub fn add<T>(&mut self, tool: T)
    where
        T: Tool<C> + 'static,
    {
        let tool_dyn = ToolWrapper(tool, PhantomData);
        self.tools.insert(T::NAME.to_string(), Box::new(tool_dyn));
    }

    pub fn call<'a>(
        &'a self,
        toolname: &str,
        ctx: &'a C,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, BoxError>> + Send + 'a>> {
        if let Some(tool) = self.tools.get(toolname) {
            tool.call(ctx, args)
        } else {
            Box::pin(futures::future::ready(Err(
                Box::new(ToolSetError::ToolNotFoundError(toolname.to_string())) as BoxError,
            )))
        }
    }
}
