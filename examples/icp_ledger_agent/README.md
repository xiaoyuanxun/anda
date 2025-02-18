# `icp_ledger_agent`

An AI agent example to interact with ICP blockchain ledgers.

## Running locally

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
cp example.env .env
# update .env
cargo run -p icp_ledger_agent
```

Build CLI tool and run agent:
```sh
cargo build -p anda_engine_cli
./target/debug/anda_engine_cli agent-run -p 'Please check my PANDA balance'
./target/debug/anda_engine_cli tool-call -n icp_ledger_balance_of -a '{"account":"535yc-uxytb-gfk7h-tny7p-vjkoe-i4krp-3qmcl-uqfgr-cpgej-yqtjq-rqe","symbol":"PANDA"}'
```

**Notice**: The current version only supports OpenAI and Grok. Deepseek's Function Calling capabilitity is unstable. https://api-docs.deepseek.com/guides/function_calling

## License
Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../../LICENSE-MIT
