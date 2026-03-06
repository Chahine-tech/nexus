# Builder stage — compiles the Rust binary
FROM rust:bookworm AS builder
WORKDIR /build

# Pre-warm the dependency cache: copy manifests + a stub main, build deps only.
# Real source changes won't invalidate this layer.
COPY Cargo.toml Cargo.lock ./
COPY core/Cargo.toml core/Cargo.toml
RUN mkdir -p core/src && echo 'fn main() {}' > core/src/main.rs
RUN cargo build --release -p nexus-core 2>/dev/null; rm -rf core/src

# Full build with real source
COPY core/ core/
RUN touch core/src/main.rs && cargo build --release -p nexus-core

# Runtime stage — minimal image with only the binary
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/nexus /usr/local/bin/nexus

EXPOSE 3001 9000
CMD ["nexus"]
