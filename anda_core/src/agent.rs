use std::{collections::BTreeMap, future::Future, marker::PhantomData, pin::Pin};

use crate::context::{AgentContext, AgentOutput, FunctionDefinition, Value};
use crate::BoxError;

/// Core trait defining an AI agent's behavior
///
/// # Type Parameters
/// - `C`: The context type that implements `AgentContext`, must be thread-safe and have a static lifetime
pub trait Agent<C>: Send + Sync
where
    C: AgentContext + Send + Sync + 'static,
{
    /// The unique name of the agent. This name should be unique within the engine.
    ///
    /// # Rules
    /// - Must not be empty
    /// - Length must be ≤ 64 characters
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_)
    const NAME: &'static str;

    /// Returns the agent's name as a String
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Returns the agent's function definition for API integration
    ///
    /// # Returns
    /// - `FunctionDefinition`: The structured definition of the agent's capabilities
    fn definition(&self) -> FunctionDefinition;

    /// Executes the agent's main logic with given context and inputs
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing `AgentContext`
    /// - `prompt`: The input prompt or message for the agent
    /// - `attachment`: Optional additional data in JSON format
    ///
    /// # Returns
    /// - A future resolving to `Result<AgentOutput, BoxError>`
    fn run(
        &self,
        ctx: &C,
        prompt: &str,
        attachment: Option<Value>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send + Sync;
}

/// Dynamic dispatch version of Agent trait for runtime flexibility
///
/// This trait allows for runtime polymorphism of agents, enabling dynamic agent selection
/// and execution without knowing the concrete type at compile time.
pub trait AgentDyn<C>: Send + Sync
where
    C: AgentContext + Send + Sync + 'static,
{
    fn name(&self) -> String;

    fn definition(&self) -> FunctionDefinition;

    fn run<'a>(
        &'a self,
        ctx: &'a C,
        prompt: &'a str,
        attachment: Option<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput, BoxError>> + Send + 'a>>;
}

/// Adapter for converting static Agent to dynamic dispatch
struct AgentWrapper<T, C>(T, PhantomData<fn() -> C>)
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static;

impl<T, C> AgentDyn<C> for AgentWrapper<T, C>
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static,
{
    fn name(&self) -> String {
        self.0.name()
    }

    fn definition(&self) -> FunctionDefinition {
        self.0.definition()
    }

    fn run<'a>(
        &'a self,
        ctx: &'a C,
        prompt: &'a str,
        attachment: Option<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput, BoxError>> + Send + 'a>> {
        Box::pin(self.0.run(ctx, prompt, attachment))
    }
}

/// Collection of registered agents with lookup and execution capabilities
///
/// # Type Parameters
/// - `C`: The context type that implements `AgentContext`
#[derive(Default)]
pub struct AgentSet<C: AgentContext + 'static> {
    pub set: BTreeMap<String, Box<dyn AgentDyn<C>>>,
}

impl<C> AgentSet<C>
where
    C: AgentContext + Send + Sync + 'static,
{
    /// Creates a new empty AgentSet
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Checks if an agent with given name exists
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(name)
    }

    /// Retrieves definition for a specific agent
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set.get(name).map(|tool| tool.definition())
    }

    /// Returns definitions for all or specified agents
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by
    ///
    /// # Returns
    /// - `Vec<FunctionDefinition>`: Vector of agent definitions
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

    /// Registers a new agent in the set
    ///
    /// # Arguments
    /// - `agent`: The agent to register, must implement `Agent<C>`
    pub fn add<T>(&mut self, agent: T)
    where
        T: Agent<C> + 'static,
    {
        let agent_dyn = AgentWrapper(agent, PhantomData);
        self.set.insert(T::NAME.to_string(), Box::new(agent_dyn));
    }

    /// Executes a specific agent with given parameters
    ///
    /// # Arguments
    /// - `name`: The name of the agent to execute
    /// - `ctx`: The execution context
    /// - `prompt`: The input prompt
    /// - `attachment`: Optional additional data
    ///
    /// # Returns
    /// - A boxed future resolving to `Result<AgentOutput, BoxError>`
    ///
    /// # Errors
    /// - Returns an error if the agent is not found
    pub fn run<'a>(
        &'a self,
        name: &str,
        ctx: &'a C,
        prompt: &'a str,
        attachment: Option<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput, BoxError>> + Send + 'a>> {
        if let Some(agent) = self.set.get(name) {
            agent.run(ctx, prompt, attachment)
        } else {
            Box::pin(futures::future::ready(Err(format!(
                "agent {name} not found"
            )
            .into())))
        }
    }
}
