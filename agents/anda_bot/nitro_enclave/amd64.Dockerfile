# base image
FROM debian:bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y gcc g++ libc6-dev pkg-config libssl-dev wget protobuf-compiler

# working directory
WORKDIR /app

# supervisord to manage programs
RUN wget -qO- https://github.com/ochinchina/supervisord/releases/download/v0.7.3/supervisord_0.7.3_Linux_64-bit.tar.gz | tar xvz
RUN mv supervisord_0.7.3_Linux_64-bit/supervisord ./ \
    && rm -rf supervisord_0.7.3_Linux_64-bit \
    && chmod +x supervisord

# dnsproxy to provide DNS services inside the enclave
RUN wget -qO- https://github.com/AdguardTeam/dnsproxy/releases/download/v0.73.3/dnsproxy-linux-amd64-v0.73.3.tar.gz | tar xvz
RUN mv linux-amd64/dnsproxy ./ \
    && rm -rf linux-amd64 \
    && chmod +x dnsproxy

RUN wget -O ic_tee_daemon https://github.com/ldclabs/ic-tee/releases/download/v0.2.14/ic_tee_daemon
RUN chmod +x ic_tee_daemon

RUN wget -O ic_tee_nitro_gateway https://github.com/ldclabs/ic-tee/releases/download/v0.3.0/ic_tee_nitro_gateway
RUN chmod +x ic_tee_nitro_gateway

RUN wget -O anda_bot https://github.com/ldclabs/anda/releases/download/v0.4.0/anda_bot
RUN chmod +x anda_bot

FROM debian:bookworm-slim AS runtime

# install dependency tools
RUN apt-get update \
    && apt-get install -y net-tools iptables iproute2 ca-certificates tzdata openssl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app/
# working directory
WORKDIR /app

# supervisord config
COPY supervisord.conf /etc/supervisord.conf
# setup.sh script that will act as entrypoint
COPY Character.toml setup.sh ./
RUN chmod +x setup.sh && ls -la

ENV LOG_LEVEL=info

# entry point
ENTRYPOINT [ "/app/setup.sh" ]