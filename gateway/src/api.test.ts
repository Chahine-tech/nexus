import { describe, test, expect, beforeEach } from "bun:test"
import { app } from "./api"
import { registry } from "./services/NodeRegistry"

// Reset registry state before each test so tests are isolated.
beforeEach(() => {
  // Access internal nodes map via a fresh registry — we clear the shared singleton.
  // Cast to access private field for test setup only.
  const r = registry as unknown as { nodes: Map<unknown, unknown> }
  r.nodes.clear()
})

describe("GET /health", () => {
  test("returns status ok", async () => {
    const res = await app.handle(new Request("http://localhost/health"))
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body).toEqual({ status: "ok" })
  })
})

describe("POST /crawl", () => {
  test("with no live nodes returns triggered=0 and total=0", async () => {
    const res = await app.handle(
      new Request("http://localhost/crawl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seeds: ["https://example.com"] }),
      }),
    )
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body).toMatchObject({ triggered: 0, total: 0 })
  })
})

describe("GET /stats", () => {
  test("with no live nodes returns zeros", async () => {
    const res = await app.handle(new Request("http://localhost/stats"))
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body).toMatchObject({
      total_docs: 0,
      total_vocab: 0,
      pagerank_ready: false,
      live_nodes: 0,
    })
  })
})
