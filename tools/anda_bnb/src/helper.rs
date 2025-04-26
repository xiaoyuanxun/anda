use alloy::primitives::{U256, utils::format_units};
use anda_core::BoxError;

/// Helper function to parse the balance from U256 to f64
pub(crate) fn get_balance(balance: U256) -> Result<f64, BoxError> {
    let balance = format_units(balance, 18)?;
    let balance = balance.parse::<f64>()?;
    Ok(balance)
}
