# `anda_bot` 部署指南

## 本地运行

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
mkdir -p object_store
cp example.env .env
# update .env
cargo run -p anda_bot -- start-local
```

## 在 Linux 中运行非 TEE 版本

1. 安装 Rust 开发环境。
2. 安装 IC TEE CLI 工具。
   使用 rust cargo 安装:
   ```sh
   cargo install ic_tee_cli
   ic_tee_cli --help
   ```

3. 在 linux 环境下可直接下载 anda_bot 可执行文件。
   ```sh
   wget https://github.com/ldclabs/anda/releases/download/v0.4.0/anda_bot
   chmod +x anda_bot
   ```

   其它操作系统可自行编译，参考：
   ```sh
   cargo build -p anda_bot --release
   ```

4. 下载 `Character.toml`，`Config.toml`和`Config.toml` 文件：

   - https://github.com/ldclabs/anda/blob/main/agents/anda_bot/nitro_enclave/Character.toml
   - https://github.com/ldclabs/anda/blob/main/agents/anda_bot/nitro_enclave/Config.toml

   ```sh
   wget https://raw.githubusercontent.com/ldclabs/anda/refs/heads/main/agents/anda_bot/nitro_enclave/Character.toml
   wget https://raw.githubusercontent.com/ldclabs/anda/refs/heads/main/agents/anda_bot/nitro_enclave/Config.toml
   ```

   并将其内容修改为你的配置。

5. 创建 `.env` 文件，填写以下内容：
   ```sh
   LOG_LEVEL=info
   ID_SECRET=0000000000000000000000000000000000000000000000000000000000000000
   ROOT_SECRET=000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
   CHARACTER_FILE_PATH='./Character.toml'
   CONFIG_FILE_PATH='./Config.toml'
   OBJECT_STORE_PATH='./object_store'
   ```

   使用 `ic_tee_cli` 生成你自己的 ID_SECRET 和 ROOT_SECRET：
   ```sh
   # ID_SECRET
   ic_tee_cli rand-bytes --len 32
   # ROOT_SECRET
   ic_tee_cli rand-bytes --len 48
   ```

6. 启动 anda_bot
   ```sh
   mkdir -p object_store
   ./anda_bot
   ```

## 部署 TEE 版本

### 准备环境
1. 本地安装 Docker 环境。
2. 本地安装 Rust 开发环境。
3. 本地安装 ICP 的 dfx 工具，参考：[Install Guide](https://internetcomputer.org/docs/current/developer-docs/getting-started/install)。
4. 本地安装 IC TEE CLI 工具。
   使用 rust cargo 安装:
   ```sh
   cargo install ic_tee_cli
   ic_tee_cli --help
   ```

   验证在线的 TEE attestation，如 Anda ICP 的 attestation：
   ```sh
   ic_tee_cli tee-verify --url https://andaicp.anda.bot/.well-known/attestation
   ```

   生成专用 ID:
   ```sh
   ic_tee_cli identity-new --path myid.pem
   # principal: vh57d-**-nqe

   ic_tee_cli -i myid.pem
   ```
   你需要把该 principal ID 添加到 IC COSE 服务你的 namespace 的 managers 中。

### 准备资源
#### 复制 anda_bot 部署文件。
```sh
git clone --depth 1 https://github.com/ldclabs/anda.git
cp -r anda/agents/anda_bot/nitro_enclave my_bot
cd my_bot
```

#### IC COSE 服务
IC COSE 服务是部署在 ICP 区块链上的智能合约服务，用于存储配置文件、派生密钥、派生固定身份ID等。
详情请见：[IC COSE](https://github.com/ldclabs/ic-cose)。

IC COSE 是多租户服务，以 namespace 作为管理单位，不同 namespace 之间的数据和权限完全隔离，IC COSE 智能合约的控制者对 namespace 也没有任何控制权限，除非被授权。

你可以使用 ICPanda DAO 提供的 [IC COSE service](https://dashboard.internetcomputer.org/canister/53cyg-yyaaa-aaaap-ahpua-cai)，或者自己部署一个 IC COSE 服务。

如果使用 ICPanda DAO 提供的 IC COSE 服务，请提供你期望的 namespace 名称以及管理员 principal ID，联系我们创建（目前免费）。

自己部署 IC COSE 服务请参考 https://github.com/ldclabs/ic-cose/tree/main/src/ic_cose_canister。

创建 namespace：
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

将 IC TEE CLI 的 principal ID 添加为 manager：
```sh
dfx canister call ic_cose_canister namespace_add_managers '("my_namespace", vec { principal "vh57d-**-nqe" })' --ic
```

可以查询 `my_namespace` namespace 下的任意 name 的固定身份ID，比如查询 `mybot`：
```sh
dfx canister call ic_cose_canister namespace_get_fixed_identity '("my_namespace", "mybot")' --ic
# "3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae"
```

现在需要把 `mybot` 添加为自己的 delegator：
```sh
dfx canister call ic_cose_canister namespace_add_delegator '(record {
  ns = "my_namespace";
  name = "mybot";
  delegators = vec { principal "3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae" };
})' --ic
```

还需要把 `mybot` 添加为`my_namespace`的 user：
```sh
dfx canister call ic_cose_canister namespace_add_users '("my_namespace", vec { principal "3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae" })' --ic
```

### IC Object Store 服务
IC Object Store 服务是部署在 ICP 区块链上的智能合约服务，用于存储`anda_bot`的知识记忆和其它数据。

IC Object Store 是单租户服务，你需要部署自己的 IC Object Store 服务。
详情请见：https://github.com/ldclabs/ic-cose/blob/main/src/ic_object_store_canister/README.md。

部署完成后，还需要把 `mybot` 添加为 IC Object Store 的控制人：
```sh
dfx canister update-settings --add-controller 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae ic_object_store_canister --ic
```

### TEE 服务器
在 AWS 开通一台支持 Nitro Enclave 的实例，并安装 Docker 和 Nitro Enclave 工具。由于 Anda bot 需要 4 核 CPU（lancedb 在 2 核 CPU 环境下将会告警），推荐使用 c5a.2xlarge 8核16G实例。
你还需要学习 nitro-cli 的一些基本操作，参考：https://docs.aws.amazon.com/enclaves/latest/user/getting-started.html。

由于安装了 Docker，会影响 iptables，需要清空 iptables 规则，每次重启该实例都需要清理：
```sh
sudo iptables -F
sudo iptables -t nat -F
```

另外还需要给 TEE 分配资源，我们将 cpu-count 设置为 4，memory 设置为 12000：
```sh
sudo nano /etc/nitro_enclaves/allocator.yaml
sudo systemctl enable nitro-enclaves-allocator
sudo systemctl start nitro-enclaves-allocator
```

下载 `ic_tee_host_daemon` 并启动（参考：https://github.com/ldclabs/ic-tee/tree/main/src/ic_tee_host_daemon）：
```sh
wget https://github.com/ldclabs/ic-tee/releases/download/v0.2.14/ic_tee_host_daemon
chmod +x ic_tee_host_daemon
sudo nohup ./ic_tee_host_daemon > tee.log 2>&1 &
```

### 修改 supervisord 配置文件
修改 `my_bot/supervisord.conf` 文件，`[program:ic_tee_nitro_gateway]` 的参数如下：
1. `--cose-canister` 参数填写 IC COSE 服务的 canister id。
2. `--cose-namespace` 参数填写 IC COSE 服务创建的 namespace `my_namespace`。
3. `--cose-identity-name` 参数填写 IC COSE 服务的 namespace 中设置的 identity name `mybot`。
4. `--app-basic-token` 参数是一个随机字符串。
其它参数可以根据需要修改。

然后再修改 `[program:anda_bot]` 的参数如下：
1. `--cose-canister` 与上面一致。
2. `--cose-namespace` 与上面一致。
3. `--basic-token` 与上面的`--app-basic-token`一致。
其它参数可以根据需要修改。

### 修改 anda_bot 配置文件
修改 `my_bot/Config.toml` 文件。注意，`Config.toml` 中包含了机密信息，不会打包到镜像文件中，而是通过 IC TEE CLI 加密上传到 IC COSE 服务中。`object_store_canister` 参数填写 IC Object Store 服务的 canister id。其它参数请看相关说明。

通过 IC TEE CLI 加密上传到 IC COSE 服务中（假设为 53cyg-yyaaa-aaaap-ahpua-cai），CLI 工具会自动处理加解密：
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-upsert-file --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --file Config.toml --key mybot --version 0
```

如果更新了 Config.toml 文件，需要重新上传，请注意 `version`，每次更新会 +1：
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-upsert-file --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --file Config.toml --key mybot --version 1
```

查看配置文件：
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-get --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --key mybot
```

### 准备 HTTPs 证书

另外我们还需准备访问域名和 HTTPs 证书，并把 HTTPs 证书加密上传到 IC COSE 服务中，从而响应外部的 HTTP 请求，比如验证 TEE attestation：
```sh
ic_tee_cli --ic -i myid.pem -c 53cyg-yyaaa-aaaap-ahpua-cai setting-save-tls --ns my_namespace --subject 3xdsu-akbd3-ss353-shfnt-u2ytc-qazai-yoikt-wi4ai-gpqds-7izg5-tae --key-file path_to_tls_key.key --cert-file path_to_tls_fullchain.cer
```

### 修改角色属性 `my_bot/Character.toml`
1. 修改 `my_bot/Character.toml` 文件，变成你期望的角色性格，可以借助 DeepSeek 优化角色。
2. 注意 `username` 字段用于定义存储空间、派生密钥等，一旦上线不可修改，其它字段可以修改。

### 构建 docker 镜像并上传到 TEE 服务器：
```sh
docker build -f amd64.Dockerfile -t my_bot:latest .
# docker run --entrypoint=sh -ti --rm my_bot:latest
docker save -o my_bot.tar my_bot:latest
gzip my_bot.tar # my-app.tar.gz
scp my_bot.tar.gz your-username@remote-server:/path/to/destination
```

### 在 TEE 服务器上启动：

解压并导入镜像：
```sh
gzip -d my_bot.tar.gz
docker load -i my_bot.tar
```

构建 enclave image：
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
构建完成后会打印镜像信息，得到一个 enclave image 文件 `my_bot.eif`。

使用  IC TEE CLI 工具生成当前版本 TEE 镜像的 principal ID：
```sh
ic_tee_cli -c e7tgb-6aaaa-aaaap-akqfa-cai identity-derive --seed 5007e8a48419d8d7117591c6a3dec4e2a99e4cf8776ce492b38a516205e55cfde271964280a9af676f8c3465a6579955
# principal: 5ldvb-ed2hq-7ggwv-vehtm-qua7s-ourgh-onfkr-sm2iz-2zxpr-7sacf-mae
```
其中，e7tgb-6aaaa-aaaap-akqfa-cai 为 ic_tee_identity 的 canister ID，参考：https://github.com/ldclabs/ic-tee/blob/main/src/ic_tee_identity/README.md。
`seed` 参数填写 enclave image 的 PCR0 值。

然后把该 principal ID 添加为 IC COSE 服务你的 namespace 的 `mybot` 的 delegator：
```sh
dfx canister call ic_cose_canister namespace_add_delegator '(record {
  ns = "my_namespace";
  name = "mybot";
  delegators = vec { principal "5ldvb-ed2hq-7ggwv-vehtm-qua7s-ourgh-onfkr-sm2iz-2zxpr-7sacf-mae" };
})' --ic
```

最后启动 enclave：
```sh
sudo nitro-cli run-enclave --cpu-count 4 --memory 12000 --enclave-cid 8 --eif-path my_bot.eif
```

查看启动状态：
```sh
tail -f tee.log
```

启动成功后，你就能用 ic_tee_cli 验证 TEE attestation 了：
```sh
ic_tee_cli tee-verify --url https://YOUE_DOMAIN/.well-known/attestation
```

还可以查看其它信息，比如 Anda ICP 的信息：
- TEE attestation：https://andaicp.anda.bot/.well-known/attestation
- TEE information：https://andaicp.anda.bot/.well-known/information
- anda_bot information：https://andaicp.anda.bot/.well-known/app

### 升级
使用上面的步骤制作最新的 enclave image。

然后停止当前的 enclave 并启动新的 enclave：
```sh
sudo nitro-cli describe-enclaves
# ... i-056e1ab9a31cd77a0-enc194c72cc9f51a9c ...
sudo nitro-cli terminate-enclave --enclave-id i-056e1ab9a31cd77a0-enc194c72cc9f51a9c
sudo nitro-cli run-enclave --cpu-count 4 --memory 12000 --enclave-cid 8 --eif-path my_bot.eif
```
