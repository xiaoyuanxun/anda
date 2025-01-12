use anda_core::{
    AgentContext, AgentOutput, AgentSet, BaseContext, BoxError, CacheExpiry, CacheFeatures,
    CancellationToken, CanisterFeatures, CompletionFeatures, CompletionRequest, Embedding,
    EmbeddingFeatures, HttpFeatures, KeysFeatures, ObjectMeta, Path, PutMode, PutResult,
    StateFeatures, StoreFeatures, ToolSet, Value, VectorSearchFeatures,
};
use candid::{utils::ArgumentEncoder, CandidType, Principal};
use ciborium::from_reader;
use serde::{de::DeserializeOwned, Serialize};
use std::{future::Future, sync::Arc, time::Duration};

use super::base::BaseCtx;
use crate::{
    database::{VectorSearchFeaturesDyn, VectorStore},
    model::Model,
};

pub struct AgentCtx {
    pub(crate) base: BaseCtx,
    pub(crate) model: Model,
    pub(crate) store: VectorStore,
    pub(crate) tools: Arc<ToolSet<BaseCtx>>,
    pub(crate) agents: Arc<AgentSet<AgentCtx>>,
}

impl AgentCtx {
    pub fn new(
        base: BaseCtx,
        model: Model,
        store: VectorStore,
        tools: Arc<ToolSet<BaseCtx>>,
        agents: Arc<AgentSet<AgentCtx>>,
    ) -> Self {
        Self {
            base,
            model,
            store,
            tools,
            agents,
        }
    }

    pub fn child(&self, agent_name: &str) -> Result<Self, BoxError> {
        Ok(Self {
            base: self.base.child(format!("A:{}", agent_name))?,
            model: self.model.clone(),
            store: self.store.clone(),
            tools: self.tools.clone(),
            agents: self.agents.clone(),
        })
    }

    pub fn child_base(&self, tool_name: &str) -> Result<BaseCtx, BoxError> {
        self.base.child(format!("T:{}", tool_name))
    }

    pub fn child_with(
        &self,
        agent_name: &str,
        user: String,
        caller: Option<Principal>,
    ) -> Result<Self, BoxError> {
        Ok(Self {
            base: self
                .base
                .child_with(format!("A:{}", agent_name), user, caller)?,
            model: self.model.clone(),
            store: self.store.clone(),
            tools: self.tools.clone(),
            agents: self.agents.clone(),
        })
    }

    pub fn child_base_with(
        &self,
        tool_name: &str,
        user: String,
        caller: Option<Principal>,
    ) -> Result<BaseCtx, BoxError> {
        self.base
            .child_with(format!("T:{}", tool_name), user, caller)
    }
}

impl AgentContext for AgentCtx {
    async fn tool_call(&self, name: &str, args: &Value) -> Result<Value, BoxError> {
        if !self.tools.contains(name) {
            return Err(format!("tool {} not found", name).into());
        }

        let ctx = self.child_base(name)?;
        self.tools.call(name, &ctx, args).await
    }

    async fn remote_tool_call(
        &self,
        endpoint: &str,
        tool_name: &str,
        args: &Value,
    ) -> Result<Value, BoxError> {
        self.https_signed_rpc(endpoint, "tool_call", &(tool_name, args))
            .await
    }

    async fn agent_run(
        &self,
        name: &str,
        prompt: &str,
        attachment: Option<Value>,
    ) -> Result<AgentOutput, BoxError> {
        if !self.agents.contains(name) {
            return Err(format!("agent {} not found", name).into());
        }

        let ctx = self.child(name)?;
        self.agents.run(name, &ctx, prompt, attachment).await
    }

    async fn remote_agent_run(
        &self,
        endpoint: &str,
        agent_name: &str,
        prompt: &str,
        attachment: Option<Value>,
    ) -> Result<AgentOutput, BoxError> {
        self.https_signed_rpc(endpoint, "agent_run", &(agent_name, prompt, attachment))
            .await
    }
}

impl CompletionFeatures<BoxError> for AgentCtx {
    async fn completion(&self, req: CompletionRequest) -> Result<AgentOutput, BoxError> {
        let mut res = self.model.completion(req).await?;
        // auto call tools
        if let Some(tools) = &mut res.tool_calls {
            for tool in tools {
                if let Ok(args) = serde_json::from_str(&tool.args) {
                    if let Ok(val) = self.tool_call(&tool.id, &args).await {
                        tool.result = Some(val);
                    }
                }
            }
        }

        Ok(res)
    }
}

impl EmbeddingFeatures<BoxError> for AgentCtx {
    fn ndims(&self) -> usize {
        self.model.ndims()
    }

    async fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, BoxError> {
        self.model.embed(texts).await
    }

    async fn embed_query(&self, text: &str) -> Result<Embedding, BoxError> {
        self.model.embed_query(text).await
    }
}

impl VectorSearchFeatures<BoxError> for AgentCtx {
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T>(&self, query: &str, n: usize) -> Result<Vec<(String, T)>, BoxError>
    where
        T: DeserializeOwned,
    {
        let res = self
            .store
            .top_n(self.base.path.clone(), query.to_string(), n)
            .await?;
        Ok(res
            .into_iter()
            .filter_map(|(id, doc)| from_reader(doc.as_ref()).ok().map(|doc| (id, doc)))
            .collect())
    }

    /// Same as `top_n` but returns the document ids only.
    async fn top_n_ids(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        self.store
            .top_n_ids(self.base.path.clone(), query.to_string(), n)
            .await
    }
}

impl BaseContext for AgentCtx {
    type Error = BoxError;
}

impl StateFeatures<BoxError> for AgentCtx {
    fn user(&self) -> String {
        self.base.user()
    }

    fn caller(&self) -> Option<Principal> {
        self.base.caller()
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.base.cancellation_token()
    }

    fn time_elapsed(&self) -> Duration {
        self.base.time_elapsed()
    }

    fn unix_ms() -> u64 {
        BaseCtx::unix_ms()
    }

    fn rand_bytes<const N: usize>() -> [u8; N] {
        BaseCtx::rand_bytes()
    }
}

impl KeysFeatures<BoxError> for AgentCtx {
    /// Derives a 256-bit AES-GCM key from the given derivation path
    async fn a256gcm_key(&self, derivation_path: &[&[u8]]) -> Result<[u8; 32], BoxError> {
        self.base.a256gcm_key(derivation_path).await
    }

    /// Signs a message using Ed25519 signature scheme from the given derivation path
    async fn ed25519_sign_message(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .ed25519_sign_message(derivation_path, message)
            .await
    }

    /// Verifies an Ed25519 signature from the given derivation path
    async fn ed25519_verify(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.base
            .ed25519_verify(derivation_path, message, signature)
            .await
    }

    /// Gets the public key for Ed25519 from the given derivation path
    async fn ed25519_public_key(&self, derivation_path: &[&[u8]]) -> Result<[u8; 32], BoxError> {
        self.base.ed25519_public_key(derivation_path).await
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path
    async fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .secp256k1_sign_message_bip340(derivation_path, message)
            .await
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path
    async fn secp256k1_verify_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.base
            .secp256k1_verify_bip340(derivation_path, message, signature)
            .await
    }

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path
    async fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.base
            .secp256k1_sign_message_ecdsa(derivation_path, message)
            .await
    }

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path
    async fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.base
            .secp256k1_verify_ecdsa(derivation_path, message, signature)
            .await
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path
    async fn secp256k1_public_key(&self, derivation_path: &[&[u8]]) -> Result<[u8; 33], BoxError> {
        self.base.secp256k1_public_key(derivation_path).await
    }
}

impl StoreFeatures<BoxError> for AgentCtx {
    /// Retrieves data from storage at the specified path
    async fn store_get(&self, path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        self.base.store_get(path).await
    }

    /// Lists objects in storage with optional prefix and offset filters
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results
    /// * `offset` - Optional path to start listing from (exclude)
    async fn store_list(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> Result<Vec<ObjectMeta>, BoxError> {
        self.base.store_list(prefix, offset).await
    }

    /// Stores data at the specified path with a given write mode
    ///
    /// # Arguments
    /// * `path` - Target storage path
    /// * `mode` - Write mode (Create, Overwrite, etc.)
    /// * `val` - Data to store as bytes
    async fn store_put(
        &self,
        path: &Path,
        mode: PutMode,
        val: bytes::Bytes,
    ) -> Result<PutResult, BoxError> {
        self.base.store_put(path, mode, val).await
    }

    /// Renames a storage object if the target path doesn't exist
    ///
    /// # Arguments
    /// * `from` - Source path
    /// * `to` - Destination path
    async fn store_rename_if_not_exists(&self, from: &Path, to: &Path) -> Result<(), BoxError> {
        self.base.store_rename_if_not_exists(from, to).await
    }

    /// Deletes data at the specified path
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete
    async fn store_delete(&self, path: &Path) -> Result<(), BoxError> {
        self.base.store_delete(path).await
    }
}

impl CacheFeatures<BoxError> for AgentCtx {
    /// Checks if a key exists in the cache
    fn cache_contains(&self, key: &str) -> bool {
        self.base.cache_contains(key)
    }

    /// Gets a cached value by key, returns error if not found or deserialization fails
    async fn cache_get<T>(&self, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.base.cache_get(key).await
    }

    /// Gets a cached value or initializes it if missing
    ///
    /// If key doesn't exist, calls init function to create value and cache it
    async fn cache_get_with<T, F>(&self, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        // futures_util::pin_mut!(init);
        self.base.cache_get_with(key, init).await
    }

    /// Sets a value in cache with optional expiration policy
    async fn cache_set<T>(&self, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        self.base.cache_set(key, val).await
    }

    /// Deletes a cached value by key, returns true if key existed
    async fn cache_delete(&self, key: &str) -> bool {
        self.base.cache_delete(key).await
    }
}

impl CanisterFeatures<BoxError> for AgentCtx {
    /// Performs a query call to a canister (read-only, no state changes)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    async fn canister_query<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        self.base.canister_query(canister, method, args).await
    }

    /// Performs an update call to a canister (may modify state)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    async fn canister_update<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> Result<Out, BoxError> {
        self.base.canister_update(canister, method, args).await
    }
}

impl HttpFeatures<BoxError> for AgentCtx {
    /// Makes an HTTPs request
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    async fn https_call(
        &self,
        url: &str,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> Result<reqwest::Response, BoxError> {
        self.base.https_call(url, method, headers, body).await
    }

    /// Makes a signed HTTPs request with message authentication
    ///
    /// # Arguments
    /// * `url` - Target URL
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `message_digest` - 32-byte message digest for signing
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    async fn https_signed_call(
        &self,
        url: &str,
        method: http::Method,
        message_digest: &[u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> Result<reqwest::Response, BoxError> {
        self.base
            .https_signed_call(url, method, message_digest, headers, body)
            .await
    }

    /// Makes a signed CBOR-encoded RPC call
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to
    /// * `method` - RPC method name to call
    /// * `params` - Parameters to serialize as CBOR and send with the request
    async fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        params: impl Serialize + Send,
    ) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.base.https_signed_rpc(endpoint, method, params).await
    }
}

#[cfg(test)]
mod tests {
    use ciborium::from_reader;
    use ic_cose_types::to_cbor_bytes;
    use serde_json::json;

    #[test]
    fn json_in_cbor_works() {
        let json = json!({
            "level": "info",
            "message": "Hello, world!",
            "timestamp": "2021-09-01T12:00:00Z",
            "data": {
                "key": "value",
                "number": 42,
                "flag": true
            }
        });
        let data = to_cbor_bytes(&json);
        let val: serde_json::Value = from_reader(&data[..]).unwrap();
        assert_eq!(json, val);
    }
}
