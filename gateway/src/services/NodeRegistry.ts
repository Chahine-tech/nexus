import type { NodeId } from "../errors"

// TODO: node discovery via Kademlia DHT (week 3)
export class NodeRegistry {
  private nodes = new Map<NodeId, { url: string; dead: boolean }>()

  register(nodeId: NodeId, url: string) {
    this.nodes.set(nodeId, { url, dead: false })
  }

  locate(_terms: string[]): NodeId[] {
    return [...this.nodes.entries()]
      .filter(([, n]) => !n.dead)
      .map(([id]) => id)
  }

  markDead(nodeId: NodeId) {
    const node = this.nodes.get(nodeId)
    if (node) node.dead = true
  }

  getUrl(nodeId: NodeId): string | undefined {
    return this.nodes.get(nodeId)?.url
  }
}

export const registry = new NodeRegistry()
