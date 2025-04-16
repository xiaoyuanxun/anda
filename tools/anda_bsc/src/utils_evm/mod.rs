//! Module for interacting with BSC chain
//!
//! This module provides functionality for:
//! - Compose raw transaction data for Json Rpc api `eth_sendRawTransaction` 
use alloy::primitives::{utils::format_units, U256};
use anda_core::BoxError;

/// Helper function to parse the balance from U256
pub(crate) fn get_balance(balance: U256) -> Result<f64, BoxError> {
    let balance = format_units(balance, 18)?;
    let balance = balance.parse::<f64>()?;
    Ok(balance)
}

/// Converts a slice of byte slices into a boxed slice of boxed byte slices.
pub fn convert_to_boxed(slices: &[&[u8]]) -> Box<[Box<[u8]>]> {
    slices
        .iter()
        .map(|&slice| slice.to_vec().into_boxed_slice())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Convert a boxed slice of boxed byte slices into a Vector of byte slices.
pub fn box_to_slice<'a>(boxed_slices: &'a Box<[Box<[u8]>]>) -> Vec<&'a [u8]> {
    boxed_slices
        .iter().
        map(|x| x.as_ref())
        .collect::<Vec<_>>()
}