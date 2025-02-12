use anda_core::{cbor_rpc, BoxError, BoxPinFut, RPCRequest};
use anda_engine::context::Web3ClientFeatures;
use candid::{
    utils::{encode_args, ArgumentEncoder},
    CandidType, Decode, Principal,
};
use ed25519_consensus::SigningKey;
use ic_agent::identity::BasicIdentity;
use ic_cose::client::CoseSDK;
use ic_cose_types::{
    cose::{
        ed25519::ed25519_verify,
        k256::{secp256k1_verify_bip340, secp256k1_verify_ecdsa},
        sha3_256,
    },
    to_cbor_bytes, CanisterCaller,
};
use ic_tee_agent::http::sign_digest_to_headers;
use std::{sync::Arc, time::Duration};

pub use ic_agent::{Agent, Identity};

use crate::crypto;
use anda_engine::APP_USER_AGENT;

/// Client for interacting with outside services (includes ICP and other blockchains)
///
/// Provides cryptographic operations, canister communication, and HTTP features.
/// Manages both internal and external HTTP clients
/// with different configurations for secure communication.
#[derive(Clone)]
pub struct Client {
    outer_http: reqwest::Client,
    root_secret: [u8; 48],
    id: Principal,
    identity: Arc<dyn Identity>,
    agent: Arc<Agent>,
    cose_canister: Principal,
}

impl Client {
    /// Creates a new Client instance
    ///
    /// # Arguments
    /// * `tee_host` - Base URL of the TEE service
    /// * `basic_token` - Authentication token for TEE access
    /// * `cose_canister` - Principal of the COSE canister
    ///
    /// # Returns
    /// Configured Client instance
    pub async fn new(
        ic_host: &str,
        id_secret: [u8; 32],
        root_secret: [u8; 48],
        cose_canister: Option<Principal>,
    ) -> Result<Self, BoxError> {
        let outer_http = reqwest::Client::builder()
            .use_rustls_tls()
            .https_only(true)
            .http2_keep_alive_interval(Some(Duration::from_secs(25)))
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .http2_keep_alive_while_idle(true)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .user_agent(APP_USER_AGENT)
            .build()
            .expect("Anda web3 client should build");

        let sk = SigningKey::from(id_secret);
        let identity = Arc::new(BasicIdentity::from_signing_key(sk));
        let id = identity.sender().expect("Failed to get sender principal");
        let agent = Agent::builder()
            .with_url(ic_host)
            .with_verify_query_signatures(false)
            .with_arc_identity(identity.clone())
            .build()?;
        if ic_host.starts_with("http://") {
            agent.fetch_root_key().await?;
        }

        Ok(Self {
            outer_http,
            root_secret,
            id,
            identity,
            agent: Arc::new(agent),
            cose_canister: cose_canister.unwrap_or(Principal::anonymous()),
        })
    }

    pub fn get_principal(&self) -> Principal {
        self.id
    }
}

impl Web3ClientFeatures for Client {
    /// Derives a 256-bit AES-GCM key from the given derivation path
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    ///
    /// # Returns
    /// Result containing the derived 256-bit key or an error
    fn a256gcm_key(&self, derivation_path: &[&[u8]]) -> BoxPinFut<Result<[u8; 32], BoxError>> {
        let res = crypto::a256gcm_key(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
        );
        Box::pin(futures::future::ready(Ok(res.into_array())))
    }

    /// Signs a message using Ed25519 signature scheme
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Message to be signed
    ///
    /// # Returns
    /// Result containing the 64-byte signature or an error
    fn ed25519_sign_message(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let res = crypto::ed25519_sign_message(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
            message,
        );
        Box::pin(futures::future::ready(Ok(res.into_array())))
    }

    /// Verifies an Ed25519 signature
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Original message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// Result indicating success or failure of verification
    fn ed25519_verify(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        let res = crypto::ed25519_public_key(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
        );
        Box::pin(futures::future::ready(
            ed25519_verify(&res.0, message, signature).map_err(|e| e.into()),
        ))
    }

    /// Gets the public key for Ed25519
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    ///
    /// # Returns
    /// Result containing the 32-byte public key or an error
    fn ed25519_public_key(
        &self,
        derivation_path: &[&[u8]],
    ) -> BoxPinFut<Result<[u8; 32], BoxError>> {
        let res = crypto::ed25519_public_key(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
        );
        Box::pin(futures::future::ready(Ok(res.0.into_array())))
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Message to be signed
    ///
    /// # Returns
    /// Result containing the 64-byte signature or an error
    fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let res = crypto::secp256k1_sign_message_bip340(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
            message,
        );
        Box::pin(futures::future::ready(Ok(res.into_array())))
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Original message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// Result indicating success or failure of verification
    fn secp256k1_verify_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        let res = crypto::secp256k1_public_key(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
        );
        Box::pin(futures::future::ready(
            secp256k1_verify_bip340(res.0.as_slice(), message, signature).map_err(|e| e.into()),
        ))
    }

    /// Signs a message using Secp256k1 ECDSA signature
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Message to be signed
    ///
    /// # Returns
    /// Result containing the 64-byte signature or an error
    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> BoxPinFut<Result<[u8; 64], BoxError>> {
        let res = crypto::secp256k1_sign_message_ecdsa(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
            message,
        );
        Box::pin(futures::future::ready(Ok(res.into_array())))
    }

    /// Verifies a Secp256k1 ECDSA signature
    ///
    /// # Arguments
    /// * `derivation_path` - Additional path components for key derivation
    /// * `message` - Original message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// Result indicating success or failure of verification
    fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> BoxPinFut<Result<(), BoxError>> {
        let res = crypto::secp256k1_public_key(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
        );
        Box::pin(futures::future::ready(
            secp256k1_verify_ecdsa(res.0.as_slice(), message, signature).map_err(|e| e.into()),
        ))
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1
    ///
    /// # Arguments
    /// * `path` - Base path for key derivation
    /// * `derivation_path` - Additional path components for key derivation
    ///
    /// # Returns
    /// Result containing the 33-byte public key or an error
    fn secp256k1_public_key(
        &self,
        derivation_path: &[&[u8]],
    ) -> BoxPinFut<Result<[u8; 33], BoxError>> {
        let res = crypto::secp256k1_public_key(
            &self.root_secret,
            derivation_path.iter().map(|v| v.to_vec()).collect(),
        );
        Box::pin(futures::future::ready(Ok(res.0.into_array())))
    }

    fn canister_query_raw(
        &self,
        canister: Principal,
        method: String,
        args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        let agent = self.agent.clone();
        Box::pin(async move {
            let res = agent.query(&canister, method).with_arg(args).call().await?;
            Ok(res)
        })
    }

    fn canister_update_raw(
        &self,
        canister: Principal,
        method: String,
        args: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        let agent = self.agent.clone();
        Box::pin(async move {
            let res = agent
                .update(&canister, method)
                .with_arg(args)
                .call_and_wait()
                .await?;
            Ok(res)
        })
    }

    fn https_call(
        &self,
        url: String,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
        let outer_http = self.outer_http.clone();
        Box::pin(async move {
            if !url.starts_with("https://") {
                return Err("Invalid URL, must start with https://".into());
            }
            let mut req = outer_http.request(method, url);
            if let Some(headers) = headers {
                req = req.headers(headers);
            }
            if let Some(body) = body {
                req = req.body(body);
            }

            req.send().await.map_err(|e| e.into())
        })
    }

    fn https_signed_call(
        &self,
        url: String,
        method: http::Method,
        message_digest: [u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> BoxPinFut<Result<reqwest::Response, BoxError>> {
        let mut headers = headers.unwrap_or_default();
        if let Err(err) =
            sign_digest_to_headers(self.identity.as_ref(), &mut headers, &message_digest)
        {
            return Box::pin(futures::future::ready(Err(err.into())));
        }

        let outer_http = self.outer_http.clone();
        Box::pin(async move {
            if !url.starts_with("https://") {
                return Err("Invalid URL, must start with https://".into());
            }
            let mut req = outer_http.request(method, url);
            req = req.headers(headers);
            if let Some(body) = body {
                req = req.body(body);
            }

            req.send().await.map_err(|e| e.into())
        })
    }

    fn https_signed_rpc_raw(
        &self,
        endpoint: String,
        method: String,
        params: Vec<u8>,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        let req = RPCRequest {
            method: &method,
            params: &params.into(),
        };
        let body = to_cbor_bytes(&req);
        let digest: [u8; 32] = sha3_256(&body);
        let mut headers = http::HeaderMap::new();
        if let Err(err) = sign_digest_to_headers(self.identity.as_ref(), &mut headers, &digest) {
            return Box::pin(futures::future::ready(Err(err.into())));
        }
        let outer_http = self.outer_http.clone();
        Box::pin(async move {
            let res = cbor_rpc(&outer_http, &endpoint, &method, Some(headers), body).await?;
            Ok(res.into_vec())
        })
    }
}

/// Implements the `CoseSDK` trait for Client to enable IC-COSE canister API calls
///
/// This implementation provides the necessary interface to interact with the
/// [IC-COSE](https://github.com/ldclabs/ic-cose) canister, allowing cryptographic
/// operations through the COSE (CBOR Object Signing and Encryption) protocol.
impl CoseSDK for Client {
    fn canister(&self) -> &Principal {
        &self.cose_canister
    }
}

impl CanisterCaller for Client {
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
        let input = encode_args(args)?;
        let res = self
            .agent
            .query(canister, method)
            .with_arg(input)
            .call()
            .await?;
        let output = Decode!(res.as_slice(), Out)?;
        Ok(output)
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
        let input = encode_args(args)?;
        let res = self
            .agent
            .update(canister, method)
            .with_arg(input)
            .call_and_wait()
            .await?;
        let output = Decode!(res.as_slice(), Out)?;
        Ok(output)
    }
}
