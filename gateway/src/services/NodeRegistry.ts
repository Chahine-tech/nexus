import { Context, Layer } from "effect"
import type { NodeId } from "../errors"

export class NodeRegistry {
  private nodes = new Map<NodeId, { url: string; dead: boolean; lastSeen: number }>()
  // Tracks which terms each node is responsible for (populated by QueryPlanner fan-out).
  private termOwnership = new Map<NodeId, Set<string>>()
  // Callbacks fired when a node transitions from live → dead.
  private onDeadCallbacks: Array<(nodeId: NodeId, url: string) => void> = []

  register(nodeId: NodeId, url: string) {
    this.nodes.set(nodeId, { url, dead: false, lastSeen: Date.now() })
  }

  // Called when a Rust node boots and self-registers with the gateway.
  registerFromRust(nodeId: NodeId, url: string) {
    this.register(nodeId, url)
  }

  // Records that `nodeId` handled `terms` in a query plan.
  recordTermOwnership(nodeId: NodeId, terms: string[]) {
    let owned = this.termOwnership.get(nodeId)
    if (!owned) {
      owned = new Set()
      this.termOwnership.set(nodeId, owned)
    }
    for (const t of terms) owned.add(t)
  }

  // Returns terms last seen on this node (used during rebalancing).
  termsOf(nodeId: NodeId): string[] {
    return [...(this.termOwnership.get(nodeId) ?? [])]
  }

  locate(_terms: string[]): NodeId[] {
    return [...this.nodes.entries()]
      .filter(([, n]) => !n.dead)
      .map(([id]) => id)
  }

  // Returns all live nodes with their URLs (used by QueryPlanner for consistent hashing).
  liveNodes(): Array<{ nodeId: NodeId; url: string }> {
    return [...this.nodes.entries()]
      .filter(([, n]) => !n.dead)
      .map(([nodeId, { url }]) => ({ nodeId, url }))
  }

  // Registers a callback invoked when a node is newly marked dead.
  onNodeDead(cb: (nodeId: NodeId, url: string) => void) {
    this.onDeadCallbacks.push(cb)
  }

  markDead(nodeId: NodeId) {
    const node = this.nodes.get(nodeId)
    if (node && !node.dead) {
      node.dead = true
      for (const cb of this.onDeadCallbacks) cb(nodeId, node.url)
    }
  }

  getUrl(nodeId: NodeId): string | undefined {
    return this.nodes.get(nodeId)?.url
  }

  // Pings all registered nodes; marks those that don't respond as dead.
  async checkHeartbeats(): Promise<void> {
    const checks = [...this.nodes.entries()].map(async ([nodeId, entry]) => {
      try {
        const res = await fetch(`${entry.url}/health`, {
          signal: AbortSignal.timeout(500),
        })
        if (res.ok) {
          entry.dead = false
          entry.lastSeen = Date.now()
        } else {
          this.markDead(nodeId)
        }
      } catch {
        this.markDead(nodeId)
      }
    })
    await Promise.allSettled(checks)
  }
}

// Effect Context.Tag for DI — use NodeRegistryService.Default layer in production.
export class NodeRegistryService extends Context.Tag("NodeRegistryService")<
  NodeRegistryService,
  NodeRegistry
>() {}

export const NodeRegistryLive = Layer.succeed(NodeRegistryService, new NodeRegistry())

export const registry = new NodeRegistry()
