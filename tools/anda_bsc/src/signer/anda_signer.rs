use alloy::consensus::SignableTransaction;
use alloy::primitives::{Address, ChainId, PrimitiveSignature as Signature, B256, U256};
use alloy::signers::{self as alloy_signer, sign_transaction_with_chain_id, Result, Signer, Error};
use async_trait::async_trait;
use std::fmt;
use anda_engine::context::BaseCtx;
use anda_core::KeysFeatures;

/// Anda signer that uses a remote TEE service via a web client.
#[derive(Clone)]
pub struct AndaSigner {
    /// The derivation path
    derivation: Box<[Box<[u8]>]>,
    /// The chain ID
    chain_id: Option<ChainId>,
    /// The Ethereum address
    address: Address,
    /// The public key
    pubkey: [u8; 32],
    /// The base context for communicating with the TEE service
    client: BaseCtx, // Todo: change to &BaseCtx?
}

impl fmt::Debug for AndaSigner {  // Todo: verify the formated output
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AndaSigner")
            .field("derivation", &self.derivation)
            .field("chain_id", &self.chain_id)
            .field("address", &self.address)
            .field("pubkey", &self.pubkey) // Include `pubkey` in the debug output
            .finish()
    }
}

/// Errors thrown by [`AndaSigner`].
#[derive(Debug, thiserror::Error)]
pub enum AndaSignerError {
    /// Web client error
    #[error("web client error: {0}")]
    WebClient(String),
    
    /// Address derivation error
    #[error("address derivation error: {0}")]
    AddressDerivation(String),

    /// Public key convertion error
    #[error("public key convertion error: {0}")]
    PubKeyConvertion(String),
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl alloy::network::TxSigner<Signature> for AndaSigner {
    fn address(&self) -> Address {
        self.address
    }

    #[inline]
    async fn sign_transaction(
        &self,
        tx: &mut dyn SignableTransaction<Signature>,
    ) -> Result<Signature> {
        sign_transaction_with_chain_id!(self, tx, self.sign_hash(&tx.signature_hash()).await)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Signer for AndaSigner {
    async fn sign_hash(&self, hash: &B256) -> Result<Signature> {
        // Convert Box<[Box<[u8]>]> to &[&[u8]]
        let derivation = self.derivation
            .iter().map(|x| x.as_ref()).collect::<Vec<_>>();
        let derivation = derivation.as_slice();

        let sig_ic = self.client.secp256k1_sign_message_ecdsa(derivation, hash.as_slice())
            .await
            .map_err(|e| Error::other(AndaSignerError::WebClient(e.to_string())))?;

        let signature = Signature::new (
            U256::from_be_slice(&sig_ic[0..32]),  // r
            U256::from_be_slice(&sig_ic[32..64]), // s
            y_parity(hash.as_slice(), &sig_ic, self.pubkey.as_slice()) // Todo: Is it compatible with EIP155        
        );

        Ok(signature)
    }

    // async fn sign_message(&self, message: &[u8]) -> Result<Signature> {
    //     self.client.secp256k1_sign_message_ecdsa(&self.derivation, message)
    //         .await
    //         .map_err(|e| Error::other(AndaSignerError::WebClient(e.to_string())))
    // }

    #[inline]
    fn address(&self) -> Address {
        self.address
    }

    #[inline]
    fn chain_id(&self) -> Option<ChainId> {
        self.chain_id
    }

    #[inline]
    fn set_chain_id(&mut self, chain_id: Option<ChainId>) {
        self.chain_id = chain_id;
    }
}

alloy::network::impl_into_wallet!(AndaSigner);

impl AndaSigner {
    /// Create a new Anda signer.
    ///
    /// This will fetch the public key from the TEE service and derive the Ethereum address.
    pub async fn new(
        client: BaseCtx, // Updated type
        derivation: Box<[Box<[u8]>]>,
        chain_id: Option<ChainId>,
    ) -> Result<Self, AndaSignerError> {
            let derivation_re = derivation.clone();
            // Convert Box<[Box<[u8]>]> to &[&[u8]]
            let derivation = derivation
                .iter().map(|x| x.as_ref()).collect::<Vec<_>>();
            let derivation = derivation.as_slice();
    
        // Fetch the public key from the TEE service
        let pubkey_bytes = client.secp256k1_public_key(&derivation)
            .await
            .map_err(|e| AndaSignerError::WebClient(e.to_string()))?;
        
        // Convert the public key to an Ethereum address
        let address = derive_address_from_pubkey(&pubkey_bytes)
            .map_err(|e| AndaSignerError::AddressDerivation(e))?;
        
        Ok(Self {
            derivation: derivation_re,
            chain_id,
            address,
            pubkey: pubkey_bytes[1..].try_into().
                    map_err(|_| AndaSignerError::PubKeyConvertion("Public key length error".to_string()))?,
            client,
        })
    }

    // /// Sign a transaction using the TEE service
    // async fn sign_tx_inner(
    //     &self,
    //     tx: &mut dyn SignableTransaction<Signature>,
    // ) -> Result<Signature> {
    //     // Get the hash of the transaction
    //     let hash = tx.signature_hash();
        
    //     // Sign the hash using the TEE service
    //     self.sign_hash(&hash).await
    // }
}

/// Helper function to derive an Ethereum address from a public key
fn derive_address_from_pubkey(pubkey: &[u8]) -> Result<Address, String> {    
    let key = k256::ecdsa::VerifyingKey::from_sec1_bytes(pubkey)
        .map_err(|e| e.to_string())?;
    Ok(alloy::signers::utils::public_key_to_address(&key))
}

/// Computes the parity bit allowing to recover the public key from the signature.
///
/// # Arguments
///
/// * `prehash` - The prehash of the message.
/// * `sig` - The signature.
/// * `pubkey` - The public key.
///
/// # Returns
///
/// The parity bit.
fn y_parity(prehash: &[u8], sig: &[u8], pubkey: &[u8]) -> bool {
    use alloy::signers::k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

    let orig_key = VerifyingKey::from_sec1_bytes(pubkey).expect("failed to parse the pubkey");
    let signature = Signature::try_from(sig).unwrap();
    for parity in [0u8, 1] {
        let recid = RecoveryId::try_from(parity).unwrap();
        let recovered_key = VerifyingKey::recover_from_prehash(prehash, &signature, recid)
            .expect("failed to recover key");
        if recovered_key == orig_key {
            match parity {
                0 => return false,
                1 => return true,
                _ => unreachable!(),
            }
        }
    }

    panic!(
        "failed to recover the parity bit from a signature; sig: {}, pubkey: {}",
        hex::encode(sig),
        hex::encode(pubkey)
    )
}

pub fn convert_to_boxed(slices: &[&[u8]]) -> Box<[Box<[u8]>]> {
    slices
        .iter()
        .map(|&slice| slice.to_vec().into_boxed_slice())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}