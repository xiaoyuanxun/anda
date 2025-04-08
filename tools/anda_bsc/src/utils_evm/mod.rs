//! Module for interacting with BSC chain
//!
//! This module provides functionality for:
//! - Compose raw transaction data for Json Rpc api `eth_sendRawTransaction` 
use alloy::primitives::{utils::format_units, Address, U256};
use anda_core::{BoxError, CONTENT_TYPE_JSON};
use rlp::RlpStream;
use secp256k1::{
    ecdsa::RecoveryId, Message, PublicKey,
    Secp256k1, SecretKey, Signing, Verification
};
use keccak_hash::{keccak, keccak256, H256};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub(crate) struct TxEvm<'a> {
    pub(crate) nonce: u64,
    pub(crate) gas_price: u128,
    pub(crate) gas_limit: u64,
    pub(crate) to: &'a Address,
    pub(crate) value: U256,
    pub(crate) data: &'a Vec<u8>,
    pub(crate) v: u64, 
    pub(crate) r: U256,
    pub(crate) s: U256,
}

impl TxEvm<'_> {
    /// Encode the Tx to RLP format
    pub(crate) fn rlp_tx(&self) -> RlpStream {
        // Encode the Tx to RLP format
        let mut stream = RlpStream::new();
        stream.begin_unbounded_list();
        stream.append(&self.nonce);
        stream.append(&self.gas_price);
        stream.append(&self.gas_limit);
        let to = <[u8; 20]>::from(self.to.clone());
        stream.append(&to.to_vec());
        stream.append(&self.value.to_be_bytes_vec());
        stream.append(self.data);
        stream.append(&self.v);
        stream.append(&self.r.to_be_bytes_vec());
        stream.append(&self.s.to_be_bytes_vec());
        stream.finalize_unbounded_list();
        stream
    }

    /// Encode the Tx to RLP format then hash it
    pub(crate) fn encode_hash_raw_tx(&self) -> H256 {
        let rlp = self.rlp_tx(); // Encode the Tx to RLP format 
        hash_encoded_data(rlp)  // Hash the encoded data
    }

  /// Compose raw transaction data for Json Rpc api `eth_sendRawTransaction`
  pub(crate) fn compose_tx_raw(
    &self, sk: &SecretKey)
    -> Result<RlpStream, BoxError>
  {
      let secp = Secp256k1::new();

      // Encode the Tx to RLP format then hash it
      let tx_hash = self.encode_hash_raw_tx(); 

      // Sign the tx's hash, get the signature and recovery id
      let (recovery_id, serialized_sig) = sign_tx_evm(&secp, &sk, &tx_hash);
      
      // Get r,s,v from signature
      let (r, s, v) = get_r_s_v(recovery_id, serialized_sig, self.v)?;
      let r = U256::from_be_bytes(r);
      let s = U256::from_be_bytes(s);

      // Fill the Tx with r,s,v
      let tx_evm = TxEvm {
        r,s,v,
        ..self.clone()
      };
      let stream = tx_evm.rlp_tx();
      Ok(stream)
  }

}

/// Sign the tx's hash
pub(crate) fn sign_tx_evm<C: Signing + Verification>(
    secp: &Secp256k1<C>,
    sk: &SecretKey,
    tx_hash: &H256,
) -> (RecoveryId, [u8; 64]) {
    let msg = Message::from_digest(tx_hash.0);
    let signature = secp.sign_ecdsa_recoverable(&msg, &sk);

    // Recover public key from signature  // Todo: Add to test, verify signature by pubkey_rcv
    // let pubkey_rcv = secp.recover_ecdsa(&msg, &signature).unwrap().serialize_uncompressed();
    // println!("pubkey_rcv: {:?}", hex::encode(&pubkey_rcv));
    signature.serialize_compact()
}

/// Get r,s,v from signature
pub(crate) fn get_r_s_v(recovery_id: RecoveryId, serialized_sig: [u8; 64], chain_id: u64) 
    -> Result<([u8; 32], [u8; 32], u64), BoxError> {
        let r: [u8; 32] = serialized_sig[0..32].try_into()?;
        let s: [u8; 32] = serialized_sig[32..].try_into()?;
        let recovery_id = i32::from(recovery_id);
        let v = recovery_id as u64 + 35 + chain_id * 2;
        Ok((r, s, v))
}

/// Hash encoded RLP data
fn hash_encoded_data(stream: RlpStream) -> H256 {
    let mut out = stream.out();
    let tx_hash = out.as_mut();
    keccak::<&[u8]>(tx_hash)
}

/// Generate sepc256k1 secrete key from root secret
pub(crate) fn generate_secret_key(root_secret: &[u8]) -> Result<SecretKey, secp256k1::Error> {
    let sk = ic_ed25519::PrivateKey::generate_from_seed(root_secret);
    let sk:&[u8] = &sk.serialize_raw();
    SecretKey::from_slice(sk)
}

/// Derive EVM Address from SecretKey
#[cfg(test)]
pub(crate) fn derive_evm_address(sk: &SecretKey) -> String {
    let secp = Secp256k1::new();
    // Derive PublicKey from SecretKey
    let pk = PublicKey::from_secret_key(&secp, &sk);
    
    // Serialize the public key in uncompressed format (65 bytes, starts with 0x04)
    let mut public_key_bytes = pk.serialize_uncompressed();
    println!("User public key: {:?}", hex::encode(&public_key_bytes));
    
    // Remove the 0x04 prefix (first byte)
    let public_key_without_prefix = &mut public_key_bytes[1..];
    
    // Hash the remaining 40 bytes
    let address = &keccak::<&[u8]>(public_key_without_prefix)[12..];
    
    // let address = format!("0x{}", hex::encode(&address));
    let address = format!("0x{}", hex::encode(&address));
    println!("User EVM address: {}", address);
    address
}

pub(crate) fn get_http_header() -> http::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    let ct: http::HeaderValue = CONTENT_TYPE_JSON.parse().unwrap();
    headers.insert(http::header::CONTENT_TYPE, ct.clone());
    headers.insert(http::header::ACCEPT, ct);
    headers
}

/// Helper function to parse the balance from the JSON response
pub(crate) fn get_balance(balance: U256) -> Result<f64, BoxError> {
    let balance = format_units(balance, 18)?;
    let balance = balance.parse::<f64>()?;
    Ok(balance)
}

/// Helper struct to deserialize JSON-RPC response
#[derive(Deserialize, Debug)]
pub(crate) struct JsonRpcResponse {
    jsonrpc: String,
    id: u32,
    result: String,
}
