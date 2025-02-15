# `anda_engine_cli`

An AI agent example to interact with ICP blockchain ledgers.

## Running locally

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
cp example.env .env
# update .env
cargo build -p anda_engine_cli

./target/debug/anda_engine_cli --help
./target/debug/anda_engine_cli rand-bytes -l 48 -f hex
./target/debug/anda_engine_cli agent-run --help
./target/debug/anda_engine_cli agent-run -p 'Please check my PANDA balance'
```

## License
Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
