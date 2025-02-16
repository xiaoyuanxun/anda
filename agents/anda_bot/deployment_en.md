# `anda_bot` Deployment Guide

## Running locally

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
mkdir -p object_store
cp example.env .env
# update .env
cargo run -p anda_bot -- start-local
```

## Running Non-TEE Version on Linux

1. Install the Rust development environment.

2. Install the IC TEE CLI tool.
   Install using rust cargo:
   ```sh
   cargo install ic_tee_cli
   ic_tee_cli --help
   ```

3. Download the executable directly on Linux:
   ```sh
   wget https://github.com/ldclabs/anda/releases/download/v0.4.0/anda_bot
   chmod +x anda_bot
   ```

   For other operating systems, compile it yourself:
   ```sh
   cargo build -p anda_bot --release
   ```

4. Download `Character.toml` and `Config.toml` files:

   - https://github.com/ldclabs/anda/blob/main/agents/anda_bot/nitro_enclave/Character.toml
   - https://github.com/ldclabs/anda/blob/main/agents/anda_bot/nitro_enclave/Config.toml

   ```sh
   wget https://raw.githubusercontent.com/ldclabs/anda/refs/heads/main/agents/anda_bot/nitro_enclave/Character.toml
   wget https://raw.githubusercontent.com/ldclabs/anda/refs/heads/main/agents/anda_bot/nitro_enclave/Config.toml
   ```

   Modify their contents to fit your configuration.

5. Create a `.env` file with the following content:
   ```sh
   LOG_LEVEL=info
   ID_SECRET=0000000000000000000000000000000000000000000000000000000000000000
   ROOT_SECRET=000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
   CHARACTER_FILE_PATH='./Character.toml'
   CONFIG_FILE_PATH='./Config.toml'
   OBJECT_STORE_PATH='./object_store'
   ```

   Generate your own ID_SECRET and ROOT_SECRET using `ic_tee_cli`:
   ```sh
   # ID_SECRET
   ic_tee_cli rand-bytes --len 32
   # ROOT_SECRET
   ic_tee_cli rand-bytes --len 48
   ```

6. Start anda_bot
   ```sh
   mkdir -p object_store
   ./anda_bot
   ```

## Deploy TEE Version

### Prepare Environment
1. Install Docker locally.
2. Install Rust development environment locally.
3. Install ICP's dfx tool: [Install Guide](https://internetcomputer.org/docs/current/developer-docs/getting-started/install).
4. Install IC TEE CLI tool:
   ```sh
   cargo install ic_tee_cli
   ic_tee_cli --help
   ```

   Verify online TEE attestation (e.g., Anda ICP's attestation):
   ```sh
   ic_tee_cli tee-verify --url https://andaicp.anda.bot/.well-known/attestation
   ```

   Generate a dedicated ID:
   ```sh
   ic_tee_cli identity-new --path myid.pem
   # principal: vh57d-**-nqe

   ic_tee_cli -i myid.pem
   ```
   Add this principal ID to the managers of your namespace in the IC COSE service.

### Prepare Resources
#### Copy `anda_bot` Deployment Files
```sh
git clone --depth 1 https://github.com/ldclabs/anda.git
cp -r anda/agents/anda_bot/nitro_enclave my_bot
cd my_bot
```

#### IC COSE Service
IC COSE is a smart contract service on the ICP blockchain for storing configurations, deriving keys and fixed identity IDs. Learn more: [IC COSE](https://github.com/ldclabs/ic-cose).

It is multi-tenant, with namespaces isolating data and permissions. Use ICPanda DAO's [IC COSE service](https://dashboard.internetcomputer.org/canister/53cyg-yyaaa-aaaap-ahpua-cai) or deploy your own.

If using the IC COSE service provided by ICPanda DAO, provide your desired namespace name and admin principal ID, and contact us for creation (currently free).

To deploy your own IC COSE service, refer to: [IC COSE Deployment Guide](https://github.com/ldclabs/ic-cose/tree/main/src/ic_cose_canister).

Create a namespace:
```sh
MYID=$(dfx identity get-principal)
dfx canister call ic_cose_canister admin_create_namespace "(record {
  name = \"my_namespace\";
  desc = opt \"mybot namespace\";
  visibility = 0;
  managers = vec {principal \"$MYID\"};
  auditors = vec {};
  users = vec {};
})" --ic
```

Add IC TEE CLI's principal ID as a manager:
```sh
dfx canister call ic_cose_canister namespace_add_managers '("my_namespace", vec { principal "vh57d-**-nqe" })' --ic
```

Query fixed identity ID for `mybot`:
```sh
dfx canister call ic_cose_canister namespace_get_fixed_identity '("my_namespace", "mybot")' --ic
# "3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae"
```

Add `mybot` as a delegator:
```sh
dfx canister call ic_cose_canister namespace_add_delegator '(record {
  ns = "my_namespace";
  name = "mybot";
  delegators = vec { principal "3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae" };
})' --ic
```

Add `mybot` as a user:
```sh
dfx canister call ic_cose_canister namespace_add_users '("my_namespace", vec { principal "3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae" })' --ic
```

### IC Object Store Service
IC Object Store is a single-tenant smart contract service on ICP for storing `anda_bot` knowledge and data. Deploy your own: [IC Object Store](https://github.com/ldclabs/ic-cose/blob/main/src/ic_object_store_canister/README.md).

Add `mybot` as a controller:
```sh
dfx canister update-settings --add-controller 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae ic_object_store_canister --ic
```

### TEE Server
Set up an AWS Nitro Enclave instance with Docker and Nitro Enclave tools. Recommended: c5a.2xlarge (8 cores, 16GB RAM). Learn basics: [Nitro Enclave Guide](https://docs.aws.amazon.com/enclaves/latest/user/getting-started.html).

Clear iptables rules (required after each reboot):
```sh
sudo iptables -F
sudo iptables -t nat -F
```

Allocate resources for TEE (4 CPUs, 12000 MB memory):
```sh
sudo nano /etc/nitro_enclaves/allocator.yaml
sudo systemctl enable nitro-enclaves-allocator
sudo systemctl start nitro-enclaves-allocator
```

Download and start `ic_tee_host_daemon`:
```sh
wget https://github.com/ldclabs/ic-tee/releases/download/v0.2.14/ic_tee_host_daemon
chmod +x ic_tee_host_daemon
sudo nohup ./ic_tee_host_daemon > tee.log 2>&1 &
```

### Modify Supervisord Configuration
Update the `my_bot/supervisord.conf` file. For `[program:ic_tee_nitro_gateway]`, set the following parameters:
1. `--cose-canister`: Enter the IC COSE service's canister ID.
2. `--cose-namespace`: Enter the namespace `my_namespace` created in the IC COSE service.
3. `--cose-identity-name`: Enter the identity name `mybot` set in the IC COSE namespace.
4. `--app-basic-token`: Use a random string.
Adjust other parameters as needed.

For `[program:anda_bot]`, set:
1. `--cose-canister`: Same as above.
2. `--cose-namespace`: Same as above.
3. `--basic-token`: Same as `--app-basic-token`.
Adjust other parameters as needed.

### Modify `anda_bot` Configuration
Update the `my_bot/Config.toml` file. Note: `Config.toml` contains confidential information and is not included in the image. It is encrypted and uploaded to the IC COSE service using IC TEE CLI. Set `object_store_canister` to the IC Object Store service's canister ID. Refer to the documentation for other parameters.

Encrypt and upload `Config.toml` to the IC COSE service (e.g., 53cyg-yyaaa-aaaap-ahpua-cai):
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-upsert-file --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --file Config.toml --key mybot --version 0
```

If `Config.toml` is updated, re-upload it. Increment `version` by 1 each time:
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-upsert-file --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --file Config.toml --key mybot --version 1
```

View the configuration:
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-get --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --key mybot
```

### Prepare HTTPS Certificates
Additionally, we need to prepare the access domain and HTTPS certificates, and upload the encrypted HTTPS certificates to the IC COSE service to respond to external HTTP requests, such as verifying TEE attestation:
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-save-tls --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --key-file path_to_tls_key.key --cert-file path_to_tls_fullchain.cer
```

### Modify Role Attributes `my_bot/Character.toml`
1. Modify the `my_bot/Character.toml` file to reflect your desired role personality, which can be optimized with DeepSeek.
2. Note that the `username` field is used to define storage space, derived keys, etc., and cannot be changed once live. Other fields can be modified.

### Build and Upload Docker Image
```sh
docker build -f amd64.Dockerfile -t my_bot:latest .
# docker run --entrypoint=sh -ti --rm my_bot:latest
docker save -o my_bot.tar my_bot:latest
gzip my_bot.tar # my-app.tar.gz
scp my_bot.tar.gz your-username@remote-server:/path/to/destination
```

### Start on TEE Server
Load docker image:
```sh
gzip -d my_bot.tar.gz
docker load -i my_bot.tar
```

Build the enclave image:
```sh
sudo nitro-cli build-enclave --docker-uri my_bot:latest --output-file my_bot.eif
# Start building the Enclave Image...
# Using the locally available Docker image...
# Enclave Image successfully created.
# {
#   "Measurements": {
#     "HashAlgorithm": "Sha384 { ... }",
#     "PCR0": "5007e8a48419d8d7117591c6a3dec4e2a99e4cf8776ce492b38a516205e55cfde271964280a9af676f8c3465a6579955",
#     "PCR1": "4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493",
#     "PCR2": "ed2ca28963f4967b791ac8ef5967c8c15075e33fb9bca0904ba1e0b53bd97b0105ac87c89a110ceea7b7a466a54c3841"
#   }
# }
```
After the build is completed, the image information will be printed, and an enclave image file `my_bot.eif` will be generated.

Use the IC TEE CLI tool to generate the principal ID for the current version of the TEE image:
```sh
ic_tee_cli -c e7tgb-6aaaa-aaaap-akqfa-cai identity-derive --seed 5007e8a48419d8d7117591c6a3dec4e2a99e4cf8776ce492b38a516205e55cfde271964280a9af676f8c3465a6579955
# principal: 5ldvb-ed2hq-7ggwv-vehtm-qua7s-ourgh-onfkr-sm2iz-2zxpr-7sacf-mae
```
Here, `e7tgb-6aaaa-aaaap-akqfa-cai` is the canister ID of `ic_tee_identity`. Refer to: https://github.com/ldclabs/ic-tee/blob/main/src/ic_tee_identity/README.md.
The `seed` parameter should be filled with the PCR0 value of the enclave image.

Next, add this principal ID as a delegator for `mybot` in your namespace of the IC COSE service:
```sh
dfx canister call ic_cose_canister namespace_add_delegator '(record {
  ns = "my_namespace";
  name = "mybot";
  delegators = vec { principal "5ldvb-ed2hq-7ggwv-vehtm-qua7s-ourgh-onfkr-sm2iz-2zxpr-7sacf-mae" };
})' --ic
```

Finally, start the enclave:
```sh
sudo nitro-cli run-enclave --cpu-count 4 --memory 12000 --enclave-cid 8 --eif-path my_bot.eif
```

Check the startup status:
```sh
tail -f tee.log
```

Once successfully started, you can use `ic_tee_cli` to verify TEE attestation:
```sh
ic_tee_cli tee-verify --url https://YOUE_DOMAIN/.well-known/attestation
```

You can also view other information, such as Anda ICP's details:
- TEE attestation: https://andaicp.anda.bot/.well-known/attestation
- TEE information: https://andaicp.anda.bot/.well-known/information
- anda_bot information: https://andaicp.anda.bot/.well-known/app

### Upgrade
Follow the steps above to create the latest enclave image.

Then stop the current enclave and start the new one:
```sh
sudo nitro-cli describe-enclaves
# ... i-056e1ab9a31cd77a0-enc194c72cc9f51a9c ...
sudo nitro-cli terminate-enclave --enclave-id i-056e1ab9a31cd77a0-enc194c72cc9f51a9c
sudo nitro-cli run-enclave --cpu-count 4 --memory 12000 --enclave-cid 8 --eif-path my_bot.eif
```
