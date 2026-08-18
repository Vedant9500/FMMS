//! Fast Multipole Method (FMM) for N-body gravitational dynamics.
//!
//! Phase 0 foundation: validated math core.
//!   - Solid-harmonic expansion basis (Dehnen 2014, arXiv:1405.2255)
//!   - Morton-sorted octree with expansion centers at center of mass
//!   - Error-controlled multipole acceptance criterion, two-parameter (p, ε)
//!   - Mutual dual-tree walk (momentum conserving)
//!   - Brute-force reference for validation
//!
//! All arithmetic is scalar f64. No external dependencies.
//! SIMD / CUDA / parallelism arrive in later phases.

pub mod bodies;
pub mod direct;
pub mod fmm;
pub mod harmonic;
pub mod mac;
pub mod models;
pub mod moments;
pub mod p2p;
pub mod roofline;
pub mod rotation;
pub mod translate;
pub mod tree;
pub mod walk;

pub use fmm::Fmm;

/// Raw pointer that may cross thread boundaries; callers guarantee the
/// pointed-to accesses are disjoint per thread (level-parallel passes).
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    /// Accessor: closures must capture the whole wrapper (2021 disjoint
    /// capture would otherwise grab the raw pointer field directly).
    pub(crate) fn ptr(self) -> *mut T {
        self.0
    }
}
