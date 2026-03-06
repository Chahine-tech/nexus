# Gateway — Bun TypeScript service
FROM oven/bun:1
WORKDIR /app

# Install workspace deps (root lockfile + all workspace package.json manifests)
COPY package.json bun.lock ./
COPY gateway/package.json gateway/
# proto workspace package (optional — silently skipped if absent)
COPY proto/package.json proto/

RUN bun install --frozen-lockfile

# Copy full source
COPY gateway/ gateway/
COPY proto/ proto/

WORKDIR /app/gateway
EXPOSE 3000
CMD ["bun", "index.ts"]
