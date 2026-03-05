use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// 32-byte node identity — blake3 hash of the node's ed25519 verifying key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub [u8; 32]);

// Custom serde: serialize as msgpack bin (raw bytes), not as a sequence of 32 integers.
impl Serialize for NodeId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for NodeId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct NodeIdVisitor;
        impl<'de> serde::de::Visitor<'de> for NodeIdVisitor {
            type Value = NodeId;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "exactly 32 bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<NodeId, E> {
                <[u8; 32]>::try_from(v)
                    .map(NodeId)
                    .map_err(|_| E::invalid_length(v.len(), &self))
            }
        }
        d.deserialize_bytes(NodeIdVisitor)
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum MessageError {
    #[error("encode error: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("decode error: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
}

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    QueryRequest,
    QueryResponse,
    GossipIdf,
    GossipPagerank,
    DhtFindNode,
    DhtStore,
    Heartbeat,
    NodeJoin,
}

/// Top-level network envelope — wraps any payload with sender identity + signature.
#[must_use]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub kind: MessageType,
    /// msgpack-encoded inner payload (QueryRequest, GossipState, etc.)
    pub payload: Vec<u8>,
    pub sender: NodeId,
    #[serde(with = "signature_serde")]
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub terms: Vec<String>,
    pub limit: usize,
    pub request_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub request_id: u64,
    pub results: Vec<(u32, f32)>,
    pub node_id: NodeId,
}

// ---------------------------------------------------------------------------
// Signature serde helper — serialize [u8; 64] as msgpack bin
// ---------------------------------------------------------------------------

mod signature_serde {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        struct SigVisitor;
        impl<'de> serde::de::Visitor<'de> for SigVisitor {
            type Value = [u8; 64];
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "exactly 64 bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<[u8; 64], E> {
                <[u8; 64]>::try_from(v)
                    .map_err(|_| E::invalid_length(v.len(), &self))
            }
        }
        d.deserialize_bytes(SigVisitor)
    }
}

// ---------------------------------------------------------------------------
// Encode / decode helpers
// ---------------------------------------------------------------------------

pub fn encode_message<T: Serialize>(msg: &T) -> Result<Vec<u8>, MessageError> {
    rmp_serde::to_vec(msg).map_err(MessageError::Encode)
}

pub fn decode_message<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, MessageError> {
    rmp_serde::from_slice(bytes).map_err(MessageError::Decode)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_query_request() {
        let qr = QueryRequest {
            terms: vec!["rust".to_string(), "async".to_string()],
            limit: 10,
            request_id: 42,
        };
        let bytes = encode_message(&qr).unwrap();
        let decoded: QueryRequest = decode_message(&bytes).unwrap();
        assert_eq!(decoded.request_id, 42);
        assert_eq!(decoded.terms, qr.terms);
    }

    #[test]
    fn roundtrip_network_message() {
        let msg = NetworkMessage {
            kind: MessageType::QueryRequest,
            payload: vec![1, 2, 3],
            sender: NodeId([7u8; 32]),
            signature: [0u8; 64],
        };
        let bytes = encode_message(&msg).unwrap();
        let decoded: NetworkMessage = decode_message(&bytes).unwrap();
        assert_eq!(decoded.sender, NodeId([7u8; 32]));
        assert_eq!(decoded.payload, vec![1, 2, 3]);
    }

    #[test]
    fn node_id_roundtrip() {
        let id = NodeId([42u8; 32]);
        let bytes = encode_message(&id).unwrap();
        let decoded: NodeId = decode_message(&bytes).unwrap();
        assert_eq!(decoded, id);
    }
}
