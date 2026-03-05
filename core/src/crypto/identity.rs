use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use thiserror::Error;

use crate::network::messages::NodeId;

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("signature verification failed")]
    VerificationFailed,
    #[error("invalid key material: {0}")]
    InvalidKey(String),
}

/// ed25519 keypair for a Nexus node.
///
/// `node_id()` returns the blake3 hash of the verifying key bytes.
pub struct NodeKeypair {
    signing_key: SigningKey,
}

impl NodeKeypair {
    pub fn generate() -> Self {
        Self { signing_key: SigningKey::generate(&mut OsRng) }
    }

    /// Returns this node's identity — blake3 hash of the ed25519 verifying key.
    pub fn node_id(&self) -> NodeId {
        let pub_bytes = self.signing_key.verifying_key().to_bytes();
        NodeId(*blake3::hash(&pub_bytes).as_bytes())
    }

    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        self.signing_key.sign(msg).to_bytes()
    }

    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Verifies that `sig` was produced by the node identified by `node_id`.
    ///
    /// Two checks are performed:
    /// 1. `blake3(vk_bytes) == node_id` — the key actually belongs to this node
    /// 2. The ed25519 signature over `msg` is valid for the verifying key
    pub fn verify(
        node_id: &NodeId,
        vk_bytes: &[u8; 32],
        msg: &[u8],
        sig: &[u8; 64],
    ) -> Result<(), CryptoError> {
        // Confirm the verifying key hashes to the claimed node_id.
        let expected_id = NodeId(*blake3::hash(vk_bytes).as_bytes());
        if expected_id != *node_id {
            return Err(CryptoError::VerificationFailed);
        }

        let vk = VerifyingKey::from_bytes(vk_bytes)
            .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
        let signature = Signature::from_bytes(sig);
        vk.verify(msg, &signature).map_err(|_| CryptoError::VerificationFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_sign_verify_roundtrip() {
        let kp = NodeKeypair::generate();
        let msg = b"hello nexus";
        let sig = kp.sign(msg);
        let vk = kp.verifying_key_bytes();
        NodeKeypair::verify(&kp.node_id(), &vk, msg, &sig).unwrap();
    }

    #[test]
    fn tampered_message_fails_verify() {
        let kp = NodeKeypair::generate();
        let sig = kp.sign(b"original");
        let vk = kp.verifying_key_bytes();
        let result = NodeKeypair::verify(&kp.node_id(), &vk, b"tampered", &sig);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_node_id_fails_verify() {
        let kp1 = NodeKeypair::generate();
        let kp2 = NodeKeypair::generate();
        let msg = b"test";
        let sig = kp1.sign(msg);
        let vk1 = kp1.verifying_key_bytes();
        // kp2's node_id but kp1's verifying key — hash mismatch
        let result = NodeKeypair::verify(&kp2.node_id(), &vk1, msg, &sig);
        assert!(result.is_err());
    }
}
