use anda_core::{
    canister_rpc, cbor_rpc, http_rpc, BaseContext, BoxError, CacheExpiry, CacheFeatures,
    CancellationToken, CanisterFeatures, HttpFeatures, HttpRPCError, KeysFeatures, ObjectMeta,
    Path, PutMode, PutResult, RPCRequest, StateFeatures, StoreFeatures,
};
use candid::{utils::ArgumentEncoder, CandidType, Principal};
use ciborium::from_reader;
use ic_cose::rand_bytes;
use ic_cose_types::{cose::sha3_256, to_cbor_bytes};
use rand::Rng;
use reqwest::Client;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};
use structured_logger::unix_ms;

const CONTEXT_MAX_DEPTH: u8 = 42;
const CACHE_MAX_CAPACITY: u64 = 1000000;

use super::{cache::CacheService, keys::KeysService};
use crate::{store::Store, APP_USER_AGENT};

#[derive(Clone)]
pub struct BaseCtx {
    pub(crate) user: Option<String>,
    pub(crate) caller: Option<Principal>,
    pub(crate) path: Path,
    pub(crate) cancellation_token: CancellationToken,
    pub(crate) start_at: Instant,
    pub(crate) depth: u8,

    local_http: Client,
    outer_http: Client,
    cache: Arc<CacheService>,
    keys: Arc<KeysService>,
    store: Store,
    endpoint_identity: String,
    endpoint_canister_query: String,
    endpoint_canister_update: String,
}

impl BaseCtx {
    pub(crate) fn new(
        tee_host: &str,
        cancellation_token: CancellationToken,
        local_http: Client,
        store: Store,
    ) -> Self {
        let outer_http = reqwest::Client::builder()
            .use_rustls_tls()
            .https_only(true)
            .http2_keep_alive_interval(Some(Duration::from_secs(25)))
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .http2_keep_alive_while_idle(true)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .user_agent(APP_USER_AGENT)
            .build()
            .expect("Anda reqwest client should build");

        Self {
            user: None,
            caller: None,
            path: Path::default(),
            cancellation_token,
            start_at: Instant::now(),
            local_http: local_http.clone(),
            outer_http,
            cache: Arc::new(CacheService::new(CACHE_MAX_CAPACITY)),
            store,
            keys: Arc::new(KeysService::new(format!("{}/keys", tee_host), local_http)),
            endpoint_identity: format!("{}/identity", tee_host),
            endpoint_canister_query: format!("{}/canister/query", tee_host),
            endpoint_canister_update: format!("{}/canister/update", tee_host),
            depth: 0,
        }
    }

    pub(crate) fn child(&self, path: String) -> Result<Self, BoxError> {
        let path = Path::parse(path)?;
        let child = Self {
            path,
            cancellation_token: self.cancellation_token.child_token(),
            depth: self.depth + 1,
            ..self.clone()
        };

        if child.depth >= CONTEXT_MAX_DEPTH {
            return Err("Context depth limit exceeded".into());
        }
        Ok(child)
    }

    pub(crate) fn child_with(
        &self,
        path: String,
        user: Option<String>,
        caller: Option<Principal>,
    ) -> Result<Self, BoxError> {
        let path = Path::parse(path)?;
        let child = Self {
            path,
            user,
            caller,
            start_at: Instant::now(),
            cancellation_token: self.cancellation_token.child_token(),
            depth: self.depth + 1,
            ..self.clone()
        };

        if child.depth >= CONTEXT_MAX_DEPTH {
            return Err("Context depth limit exceeded".into());
        }
        Ok(child)
    }
}

impl BaseContext for BaseCtx {}

impl StateFeatures for BaseCtx {
    fn user(&self) -> Option<String> {
        self.user.clone()
    }

    fn caller(&self) -> Option<Principal> {
        self.caller
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    fn time_elapsed(&self) -> Duration {
        self.start_at.elapsed()
    }

    fn unix_ms() -> u64 {
        unix_ms()
    }

    /// Generates N random bytes
    fn rand_bytes<const N: usize>() -> [u8; N] {
        rand_bytes()
    }

    /// Generates a random number within the given range
    fn rand_number<T, R>(range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        let mut rng = rand::thread_rng();
        rng.gen_range(range)
    }
}

impl KeysFeatures for BaseCtx {
    /// Derives a 256-bit AES-GCM key from the given derivation path
    async fn a256gcm_key(&self, derivation_path: &[&[u8]]) -> Result<[u8; 32], BoxError> {
        self.keys.a256gcm_key(&self.path, derivation_path).await
    }

    /// Signs a message using Ed25519 signature scheme from the given derivation path
    async fn ed25519_sign_message(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.keys
            .ed25519_sign_message(&self.path, derivation_path, message)
            .await
    }

    /// Verifies an Ed25519 signature from the given derivation path
    async fn ed25519_verify(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.keys
            .ed25519_verify(&self.path, derivation_path, message, signature)
            .await
    }

    /// Gets the public key for Ed25519 from the given derivation path
    async fn ed25519_public_key(&self, derivation_path: &[&[u8]]) -> Result<[u8; 32], BoxError> {
        self.keys
            .ed25519_public_key(&self.path, derivation_path)
            .await
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path
    async fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.keys
            .secp256k1_sign_message_bip340(&self.path, derivation_path, message)
            .await
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path
    async fn secp256k1_verify_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.keys
            .secp256k1_verify_bip340(&self.path, derivation_path, message, signature)
            .await
    }

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path
    async fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        self.keys
            .secp256k1_sign_message_ecdsa(&self.path, derivation_path, message)
            .await
    }

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path
    async fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        self.keys
            .secp256k1_verify_ecdsa(&self.path, derivation_path, message, signature)
            .await
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path
    async fn secp256k1_public_key(&self, derivation_path: &[&[u8]]) -> Result<[u8; 33], BoxError> {
        self.keys
            .secp256k1_public_key(&self.path, derivation_path)
            .await
    }
}

impl StoreFeatures for BaseCtx {
    /// Retrieves data from storage at the specified path
    async fn store_get(&self, path: &Path) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        self.store.store_get(&self.path, path).await
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
        self.store.store_list(&self.path, prefix, offset).await
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
        self.store.store_put(&self.path, path, mode, val).await
    }

    /// Renames a storage object if the target path doesn't exist
    ///
    /// # Arguments
    /// * `from` - Source path
    /// * `to` - Destination path
    async fn store_rename_if_not_exists(&self, from: &Path, to: &Path) -> Result<(), BoxError> {
        self.store
            .store_rename_if_not_exists(&self.path, from, to)
            .await
    }

    /// Deletes data at the specified path
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete
    async fn store_delete(&self, path: &Path) -> Result<(), BoxError> {
        self.store.store_delete(&self.path, path).await
    }
}

impl CacheFeatures for BaseCtx {
    /// Checks if a key exists in the cache
    fn cache_contains(&self, key: &str) -> bool {
        self.cache.contains(&self.path, key)
    }

    /// Gets a cached value by key, returns error if not found or deserialization fails
    async fn cache_get<T>(&self, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        self.cache.get(&self.path, key).await
    }

    /// Gets a cached value or initializes it if missing
    ///
    /// If key doesn't exist, calls init function to create value and cache it
    async fn cache_get_with<T, F>(&self, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        self.cache.get_with(&self.path, key, init).await
    }

    /// Sets a value in cache with optional expiration policy
    async fn cache_set<T>(&self, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        self.cache.set(&self.path, key, val).await
    }

    /// Deletes a cached value by key, returns true if key existed
    async fn cache_delete(&self, key: &str) -> bool {
        self.cache.delete(&self.path, key).await
    }
}

impl CanisterFeatures for BaseCtx {
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
        let res = canister_rpc(
            &self.local_http,
            &self.endpoint_canister_query,
            canister,
            method,
            args,
        )
        .await?;
        Ok(res)
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
        let res = canister_rpc(
            &self.local_http,
            &self.endpoint_canister_update,
            canister,
            method,
            args,
        )
        .await?;
        Ok(res)
    }
}

impl HttpFeatures for BaseCtx {
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
        if !url.starts_with("https://") {
            return Err("Invalid URL, must start with https://".into());
        }
        let mut req = self.outer_http.request(method, url);
        if let Some(headers) = headers {
            req = req.headers(headers);
        }
        if let Some(body) = body {
            req = req.body(body);
        }

        req.send().await.map_err(|e| e.into())
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
        let res: HashMap<String, String> = http_rpc(
            &self.local_http,
            &self.endpoint_identity,
            "sign_http",
            &(message_digest,),
        )
        .await?;
        let mut headers = headers.unwrap_or_default();
        res.into_iter().for_each(|(k, v)| {
            headers.insert(
                http::HeaderName::try_from(k).expect("invalid header name"),
                http::HeaderValue::try_from(v).expect("invalid header value"),
            );
        });
        self.https_call(url, method, Some(headers), body).await
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
        let params = to_cbor_bytes(&params);
        let req = RPCRequest {
            method,
            params: &params.into(),
        };
        let body = to_cbor_bytes(&req);
        let digest: [u8; 32] = sha3_256(&body);
        let res: HashMap<String, String> = http_rpc(
            &self.local_http,
            &self.endpoint_identity,
            "sign_http",
            &(digest,),
        )
        .await?;
        let mut headers = http::HeaderMap::new();
        res.into_iter().for_each(|(k, v)| {
            headers.insert(
                http::HeaderName::try_from(k).expect("invalid header name"),
                http::HeaderValue::try_from(v).expect("invalid header value"),
            );
        });

        let res = cbor_rpc(&self.outer_http, endpoint, method, Some(headers), body).await?;
        let res = from_reader(&res[..]).map_err(|e| HttpRPCError::ResultError {
            endpoint: endpoint.to_string(),
            path: method.to_string(),
            error: e.into(),
        })?;
        Ok(res)
    }
}
