use serde::{de::DeserializeOwned, Serialize};
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{context::BaseContext, model::FunctionDefinition, BoxError, BoxPinFut};

/// Core trait for implementing tools that can be used by the AI Agent system
///
/// # Type Parameters
/// - `C`: The context type that implements `BaseContext`, must be thread-safe and have a static lifetime
pub trait Tool<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// The unique name of the tool. This name should be unique within the engine.
    ///
    /// # Rules
    /// - Must not be empty
    /// - Length must be ≤ 64 characters
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_)
    const NAME: &'static str;

    /// The arguments type of the tool.
    type Args: DeserializeOwned + Send + Sync;
    /// The output type of the tool.
    type Output: Serialize;

    /// Returns the tool's name
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Provides the tool's definition including its parameters schema.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The schema definition of the tool's parameters and metadata
    fn definition(&self) -> FunctionDefinition;

    /// Executes the tool with given context and arguments
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing `BaseContext`
    /// - `args`: JSON value containing the input arguments for the tool
    ///
    /// # Returns
    /// - A future resolving to a JSON value containing the tool's output
    /// - Returns `BoxError` if execution fails
    fn call(
        &self,
        ctx: C,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, BoxError>> + Send + Sync;
}

/// Dynamic dispatch version of the Tool trait
///
/// This trait allows for runtime polymorphism of tools, enabling different tool implementations
/// to be stored and called through a common interface.
pub trait ToolDyn<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// Returns the tool's name as a String
    fn name(&self) -> String;

    /// Provides the tool's definition including its parameters schema
    fn definition(&self) -> FunctionDefinition;

    /// Executes the tool with given context and arguments using dynamic dispatch
    fn call(&self, ctx: C, args: String) -> BoxPinFut<Result<String, BoxError>>;
}

/// Wrapper to convert static Tool implementation to dynamic dispatch
struct ToolWrapper<T, C>(Arc<T>, PhantomData<C>)
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

    fn definition(&self) -> FunctionDefinition {
        self.0.definition()
    }

    fn call(&self, ctx: C, args: String) -> BoxPinFut<Result<String, BoxError>> {
        let name = self.0.name();
        let tool = self.0.clone();
        Box::pin(async move {
            let args = serde_json::from_str(&args)
                .map_err(|err| format!("tool {}, invalid args: {}", name, err))?;
            let result = tool
                .call(ctx, args)
                .await
                .map_err(|err| format!("tool {}, call failed: {}", name, err))?;
            Ok(serde_json::to_string(&result)?)
        })
    }
}

/// Collection of tools that can be used by the AI Agent
///
/// # Type Parameters
/// - `C`: The context type that implements `BaseContext`, must have a static lifetime
#[derive(Default)]
pub struct ToolSet<C: BaseContext> {
    pub set: BTreeMap<String, Box<dyn ToolDyn<C>>>,
}

impl<C> ToolSet<C>
where
    C: BaseContext + Send + Sync + 'static,
{
    /// Creates a new empty ToolSet
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Checks if a tool with the given name exists in the set
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(name)
    }

    /// Gets the definition of a specific tool by name
    ///
    /// # Returns
    /// - `Some(FunctionDefinition)` if the tool exists
    /// - `None` if the tool is not found
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set.get(name).map(|tool| tool.definition())
    }

    /// Gets definitions for multiple tools, optionally filtered by names
    ///
    /// # Arguments
    /// - `names`: Optional slice of tool names to filter by. If None or empty, returns all definitions
    ///
    /// # Returns
    /// - Vector of `FunctionDefinition` for the requested tools
    pub fn definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        let names = names.unwrap_or_default();
        self.set
            .iter()
            .filter_map(|(name, tool)| {
                if names.is_empty() || names.contains(&name.as_str()) {
                    Some(tool.definition())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Adds a new tool to the set
    ///
    /// # Arguments
    /// - `tool`: The tool to add, must implement the `Tool` trait
    pub fn add<T>(&mut self, tool: T)
    where
        T: Tool<C> + Send + Sync + 'static,
    {
        let tool_dyn = ToolWrapper(Arc::new(tool), PhantomData);
        self.set.insert(T::NAME.to_string(), Box::new(tool_dyn));
    }

    /// Calls a tool by name with the given context and arguments
    ///
    /// # Arguments
    /// - `name`: The name of the tool to call
    /// - `ctx`: The execution context
    /// - `args`: JSON value containing the input arguments
    ///
    /// # Returns
    /// - A future resolving to the tool's output as a JSON value
    /// - Returns an error if the tool is not found
    pub fn call(&self, name: &str, ctx: C, args: String) -> BoxPinFut<Result<String, BoxError>> {
        if let Some(tool) = self.set.get(name) {
            tool.call(ctx, args)
        } else {
            Box::pin(futures::future::ready(Err(format!(
                "tool {name} not found"
            )
            .into())))
        }
    }
}
