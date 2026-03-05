import { test, expect, describe } from "bun:test"
import { NodeRegistry } from "./NodeRegistry"

describe("NodeRegistry", () => {
  test("registerFromRust adds node to liveNodes", () => {
    const reg = new NodeRegistry()
    reg.registerFromRust("aabbcc", "http://localhost:3001")
    expect(reg.liveNodes()).toHaveLength(1)
    expect(reg.liveNodes()[0]).toEqual({ nodeId: "aabbcc", url: "http://localhost:3001" })
  })

  test("markDead removes node from liveNodes", () => {
    const reg = new NodeRegistry()
    reg.registerFromRust("aabbcc", "http://localhost:3001")
    reg.markDead("aabbcc")
    expect(reg.liveNodes()).toHaveLength(0)
  })

  test("multiple nodes registered independently", () => {
    const reg = new NodeRegistry()
    reg.registerFromRust("node1", "http://localhost:3001")
    reg.registerFromRust("node2", "http://localhost:3002")
    expect(reg.liveNodes()).toHaveLength(2)
  })

  test("marking one dead does not affect others", () => {
    const reg = new NodeRegistry()
    reg.registerFromRust("node1", "http://localhost:3001")
    reg.registerFromRust("node2", "http://localhost:3002")
    reg.markDead("node1")
    const live = reg.liveNodes()
    expect(live).toHaveLength(1)
    expect(live[0]!.nodeId).toBe("node2")
  })

  test("getUrl returns the correct URL", () => {
    const reg = new NodeRegistry()
    reg.registerFromRust("abc", "http://example.com")
    expect(reg.getUrl("abc")).toBe("http://example.com")
    expect(reg.getUrl("unknown")).toBeUndefined()
  })

  test("checkHeartbeats marks unreachable node as dead", async () => {
    const reg = new NodeRegistry()
    // Register a node on a port nothing is listening on.
    reg.registerFromRust("dead-node", "http://127.0.0.1:19999")
    await reg.checkHeartbeats()
    expect(reg.liveNodes()).toHaveLength(0)
  })
})
