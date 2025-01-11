use anda_core::{http_rpc, BoxError};
use ic_cose_types::cose::ed25519::ed25519_verify;
use ic_cose_types::cose::k256::{secp256k1_verify_bip340, secp256k1_verify_ecdsa};
use object_store::path::Path;
use reqwest::Client;
use serde_bytes::ByteArray;

#[derive(Debug, Clone)]
pub struct KeysService {
    endpoint: String,
    http: Client,
}

impl KeysService {
    pub fn new(endpoint: String, http: Client) -> Self {
        Self { endpoint, http }
    }
}

impl KeysService {
    /// Derives a 256-bit AES-GCM key from the given derivation path
    pub async fn a256gcm_key(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
    ) -> Result<[u8; 32], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<32> =
            http_rpc(&self.http, &self.endpoint, "a256gcm_key", &(dp,)).await?;
        Ok(res.into_array())
    }

    /// Signs a message using Ed25519 signature scheme from the given derivation path
    pub async fn ed25519_sign_message(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<64> = http_rpc(
            &self.http,
            &self.endpoint,
            "ed25519_sign_message",
            &(dp, message),
        )
        .await?;
        Ok(res.into_array())
    }

    /// Verifies an Ed25519 signature from the given derivation path
    pub async fn ed25519_verify(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        let pk = self.ed25519_public_key(path, derivation_path).await?;
        ed25519_verify(&pk, message, signature).map_err(|e| e.into())
    }

    /// Gets the public key for Ed25519 from the given derivation path
    pub async fn ed25519_public_key(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
    ) -> Result<[u8; 32], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: (ByteArray<32>, ByteArray<32>) =
            http_rpc(&self.http, &self.endpoint, "ed25519_public_key", &(dp,)).await?;
        Ok(res.0.into_array())
    }

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path
    pub async fn secp256k1_sign_message_bip340(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<64> = http_rpc(
            &self.http,
            &self.endpoint,
            "secp256k1_sign_message_bip340",
            &(dp, message),
        )
        .await?;
        Ok(res.into_array())
    }

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path
    pub async fn secp256k1_verify_bip340(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        let pk = self.secp256k1_public_key(path, derivation_path).await?;
        secp256k1_verify_bip340(&pk, message, signature).map_err(|e| e.into())
    }

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path
    pub async fn secp256k1_sign_message_ecdsa(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> Result<[u8; 64], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: ByteArray<64> = http_rpc(
            &self.http,
            &self.endpoint,
            "secp256k1_sign_message_ecdsa",
            &(dp, message),
        )
        .await?;
        Ok(res.into_array())
    }

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path
    pub async fn secp256k1_verify_ecdsa(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), BoxError> {
        let pk = self.secp256k1_public_key(path, derivation_path).await?;
        secp256k1_verify_ecdsa(&pk, message, signature).map_err(|e| e.into())
    }

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path
    pub async fn secp256k1_public_key(
        &self,
        path: &Path,
        derivation_path: &[&[u8]],
    ) -> Result<[u8; 33], BoxError> {
        let mut dp = Vec::with_capacity(derivation_path.len() + 1);
        dp.push(path.as_ref().as_bytes());
        dp.extend(derivation_path);
        let res: (ByteArray<33>, ByteArray<32>) =
            http_rpc(&self.http, &self.endpoint, "secp256k1_public_key", &(dp,)).await?;
        Ok(res.0.into_array())
    }
}
