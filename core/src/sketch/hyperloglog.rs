use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// HyperLogLog — cardinality estimation for distributed IDF
// ---------------------------------------------------------------------------
//
// b=6 → m=64 registers, ~3.25% relative standard error.
// Each register stores the maximum rho (position of leftmost 1-bit, 1-based).
// Hash function: blake3 (already in Cargo), first 8 bytes used as u64.
//
// CRDT property: merge = per-register max (commutative, associative, idempotent).
//
// Serde note: serde only supports fixed arrays up to [T; 32]. Since M=64,
// we use a custom module that serializes registers as msgpack bin (raw bytes).

const B: u32 = 6;
const M: usize = 1 << B; // 64 registers

#[derive(Debug, Clone, PartialEq)]
pub struct HyperLogLog {
    registers: [u8; M],
}

// Custom serde — serialize registers as raw bytes (msgpack bin), not as a sequence of u8.
impl Serialize for HyperLogLog {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.registers)
    }
}

impl<'de> Deserialize<'de> for HyperLogLog {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct HllVisitor;
        impl<'de> serde::de::Visitor<'de> for HllVisitor {
            type Value = HyperLogLog;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "exactly {M} bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<HyperLogLog, E> {
                <[u8; M]>::try_from(v)
                    .map(|registers| HyperLogLog { registers })
                    .map_err(|_| E::invalid_length(v.len(), &self))
            }
        }
        d.deserialize_bytes(HllVisitor)
    }
}

impl HyperLogLog {
    pub fn new() -> Self {
        Self { registers: [0u8; M] }
    }

    /// Adds a value to the sketch.
    ///
    /// Uses the first 8 bytes of the blake3 hash as a u64.
    /// Top B bits select the register; remaining bits determine rho.
    pub fn add(&mut self, value: &[u8]) {
        let hash = blake3::hash(value);
        let bytes: &[u8; 32] = hash.as_bytes();
        let w = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]);

        let register_index = (w >> (64 - B)) as usize; // top 6 bits → 0..63
        let remaining = w << B;
        let rho = remaining.leading_zeros() + 1; // 1-based leftmost 1-bit position

        self.registers[register_index] = self.registers[register_index].max(rho as u8);
    }

    /// Estimates the number of distinct values added.
    ///
    /// Applies small-range (linear counting) and large-range corrections per
    /// the Flajolet et al. 2007 paper.
    pub fn estimate(&self) -> f64 {
        let alpha_64 = 0.7213 / (1.0 + 1.079 / M as f64);

        // Raw harmonic mean estimate.
        let z: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum::<f64>()
            .recip();
        let mut e = alpha_64 * (M as f64).powi(2) * z;

        // Small range correction: use linear counting when estimate is low and
        // there are empty registers.
        if e < 2.5 * M as f64 {
            let v = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if v > 0.0 {
                e = (M as f64) * (M as f64 / v).ln();
            }
        }

        // Large range correction: adjust for hash collision probability.
        let two32 = (u32::MAX as f64) + 1.0; // 2^32
        if e > two32 / 30.0 {
            e = -two32 * (1.0 - e / two32).ln();
        }

        e
    }

    /// Returns a new HLL that is the CRDT merge of self and other.
    ///
    /// Merge = per-register max. Commutative, associative, idempotent.
    pub fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.merge_in_place(other);
        result
    }

    /// Merges `other` into self in place.
    pub fn merge_in_place(&mut self, other: &Self) {
        for i in 0..M {
            self.registers[i] = self.registers[i].max(other.registers[i]);
        }
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_hll_estimates_zero() {
        let hll = HyperLogLog::new();
        assert_eq!(hll.estimate(), 0.0);
    }

    #[test]
    fn estimate_1000_distinct_items_within_15_percent() {
        let mut hll = HyperLogLog::new();
        for i in 0u32..1000 {
            hll.add(&i.to_le_bytes());
        }
        let est = hll.estimate();
        let error = ((est - 1000.0) / 1000.0).abs();
        assert!(
            error < 0.15,
            "estimate {est:.1} is more than 15% off from 1000 (error={error:.3})"
        );
    }

    #[test]
    fn merge_two_hlls_estimates_combined() {
        let mut a = HyperLogLog::new();
        let mut b = HyperLogLog::new();

        for i in 0u32..500 {
            a.add(&i.to_le_bytes());
        }
        for i in 500u32..1000 {
            b.add(&i.to_le_bytes());
        }

        let merged = a.merge(&b);
        let est = merged.estimate();
        let error = ((est - 1000.0) / 1000.0).abs();
        assert!(
            error < 0.15,
            "merged estimate {est:.1} is more than 15% off from 1000 (error={error:.3})"
        );
    }

    #[test]
    fn merge_is_idempotent() {
        let mut hll = HyperLogLog::new();
        for i in 0u32..200 {
            hll.add(&i.to_le_bytes());
        }
        let cloned = hll.clone();
        let merged = hll.merge(&cloned);
        assert_eq!(hll.estimate(), merged.estimate());
    }

    #[test]
    fn merge_is_commutative() {
        let mut a = HyperLogLog::new();
        let mut b = HyperLogLog::new();
        for i in 0u32..300 {
            a.add(&i.to_le_bytes());
        }
        for i in 300u32..600 {
            b.add(&i.to_le_bytes());
        }
        let ab = a.merge(&b).estimate();
        let ba = b.merge(&a).estimate();
        // Both should be identical (same per-register max regardless of order).
        assert!(
            (ab - ba).abs() < f64::EPSILON,
            "merge not commutative: ab={ab}, ba={ba}"
        );
    }

    #[test]
    fn serde_roundtrip() {
        let mut hll = HyperLogLog::new();
        for i in 0u32..100 {
            hll.add(&i.to_le_bytes());
        }
        let bytes = rmp_serde::to_vec(&hll).unwrap();
        let decoded: HyperLogLog = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(hll, decoded);
        assert!((hll.estimate() - decoded.estimate()).abs() < f64::EPSILON);
    }
}