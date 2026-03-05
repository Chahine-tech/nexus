import type { NodeId } from "../errors"

export class NodeRegistry {
  private nodes = new Map<NodeId, { url: string; dead: boolean; lastSeen: number }>()

  register(nodeId: NodeId, url: string) {
    this.nodes.set(nodeId, { url, dead: false, lastSeen: Date.now() })
  }

  // Called when a Rust node boots and self-registers with the gateway.
  registerFromRust(nodeId: NodeId, url: string) {
    this.register(nodeId, url)
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

  markDead(nodeId: NodeId) {
    const node = this.nodes.get(nodeId)
    if (node) node.dead = true
  }

  getUrl(nodeId: NodeId): string | undefined {
    return this.nodes.get(nodeId)?.url
  }

  // Pings all registered nodes; marks those that don't respond as dead.
  async checkHeartbeats(): Promise<void> {
    const checks = [...this.nodes.entries()].map(async ([, entry]) => {
      try {
        const res = await fetch(`${entry.url}/health`, {
          signal: AbortSignal.timeout(500),
        })
        if (res.ok) {
          entry.dead = false
          entry.lastSeen = Date.now()
        } else {
          entry.dead = true
        }
      } catch {
        entry.dead = true
      }
    })
    await Promise.allSettled(checks)
  }
}

export const registry = new NodeRegistry()
