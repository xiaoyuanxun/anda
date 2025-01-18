use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, Embedding,
    EmbeddingFeatures, ToolCall,
};
use std::sync::Arc;

pub mod cohere;
pub mod deepseek;
pub mod openai;

pub trait CompletionFeaturesDyn: Send + Sync + 'static {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>>;
}

pub trait EmbeddingFeaturesDyn: Send + Sync + 'static {
    fn ndims(&self) -> usize;

    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>>;

    fn embed_query(&self, text: String) -> BoxPinFut<Result<Embedding, BoxError>>;
}

/// A placeholder for not implemented features.
#[derive(Clone, Debug)]
pub struct NotImplemented;

impl CompletionFeaturesDyn for NotImplemented {
    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

impl EmbeddingFeaturesDyn for NotImplemented {
    fn ndims(&self) -> usize {
        0
    }

    fn embed(&self, _texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn embed_query(&self, _text: String) -> BoxPinFut<Result<Embedding, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

#[derive(Clone, Debug)]
pub struct MockImplemented;

impl CompletionFeaturesDyn for MockImplemented {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: req.prompt.clone(),
            tool_calls: if req.tools.is_empty() {
                None
            } else {
                Some(
                    req.tools
                        .iter()
                        .map(|tool| ToolCall {
                            id: tool.name.clone(),
                            name: tool.name.clone(),
                            args: req.prompt.clone(),
                            result: None,
                        })
                        .collect(),
                )
            },
            ..Default::default()
        })))
    }
}

impl EmbeddingFeaturesDyn for MockImplemented {
    fn ndims(&self) -> usize {
        384 // EMBED_MULTILINGUAL_LIGHT_V3
    }

    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>> {
        Box::pin(futures::future::ready(Ok(texts
            .into_iter()
            .map(|text| Embedding {
                text,
                vec: vec![0.0; 384],
            })
            .collect())))
    }

    fn embed_query(&self, _text: String) -> BoxPinFut<Result<Embedding, BoxError>> {
        Box::pin(futures::future::ready(Ok(Embedding {
            text: "test".to_string(),
            vec: vec![0.0; 384],
        })))
    }
}

#[derive(Clone)]
pub struct Model {
    pub embedder: Arc<dyn EmbeddingFeaturesDyn>,
    pub completer: Arc<dyn CompletionFeaturesDyn>,
}

impl Model {
    pub fn new(
        embedder: Arc<dyn EmbeddingFeaturesDyn>,
        completer: Arc<dyn CompletionFeaturesDyn>,
    ) -> Self {
        Self {
            embedder,
            completer,
        }
    }

    pub fn not_implemented() -> Self {
        Self {
            embedder: Arc::new(NotImplemented),
            completer: Arc::new(NotImplemented),
        }
    }

    pub fn mock_implemented() -> Self {
        Self {
            embedder: Arc::new(MockImplemented),
            completer: Arc::new(MockImplemented),
        }
    }
}

impl CompletionFeatures for Model {
    async fn completion(&self, req: CompletionRequest) -> Result<AgentOutput, BoxError> {
        self.completer.completion(req).await
    }
}

impl EmbeddingFeatures for Model {
    fn ndims(&self) -> usize {
        self.embedder.ndims()
    }

    async fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, BoxError> {
        self.embedder.embed(texts.into_iter().collect()).await
    }

    async fn embed_query(&self, text: &str) -> Result<Embedding, BoxError> {
        self.embedder.embed_query(text.to_string()).await
    }
}
