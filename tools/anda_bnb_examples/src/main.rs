mod balance_test;
mod transfer_test;

// use balance_test::test_bnb_ledger_balance;
use transfer_test::test_bnb_ledger_transfer;

#[tokio::main]
async fn main() {
    println!("Anda BNB test!");
    // test_bnb_ledger_balance().await;
    test_bnb_ledger_transfer().await;
}

