//! Mutual dual tree walk (Dehnen 2002).
//!
//! Starting from (root, root), a cell pair (A, B) is either
//!   - accepted on both sides -> mutual M2L (momentum-conserving);
//!   - split into the daughters of the larger cell;
//!   - computed directly (P2P) when both are leaves.
//!
//! P2P writes straight into `Bodies` (SoA slots, mutual/antisymmetric).
//! M2L dispatches to the rotation kernel for p >= ROT_P when a `RotCtx`
//! is supplied, else the naive O(p⁴) kernel.

use crate::harmonic::C;
use crate::mac::accept;
use crate::rotation::RotCtx;
use crate::translate::m2l_mutual;
use crate::tree::Octree;

use std::sync::atomic::{AtomicU64, Ordering};
static N_POP: AtomicU64 = AtomicU64::new(0);
static N_ACCEPT: AtomicU64 = AtomicU64::new(0);
static N_M2L: AtomicU64 = AtomicU64::new(0);
static N_P2P_SELF: AtomicU64 = AtomicU64::new(0);
static N_P2P_PAIR: AtomicU64 = AtomicU64::new(0);
static N_M2L_OPS: AtomicU64 = AtomicU64::new(0);
/// Reset walk counters; returns previous counts for printing.
pub fn walk_counts() -> (u64, u64, u64, u64, u64, u64) {
    (
        N_POP.swap(0, Ordering::Relaxed),
        N_ACCEPT.swap(0, Ordering::Relaxed),
        N_M2L.swap(0, Ordering::Relaxed),
        N_P2P_SELF.swap(0, Ordering::Relaxed),
        N_P2P_PAIR.swap(0, Ordering::Relaxed),
        N_M2L_OPS.swap(0, Ordering::Relaxed),
    )
}

/// Perform the dual tree walk, filling `locals` (from M2L) and updating
/// `bodies` in place (near-field P2P). The far-field potential is added
/// later by L2L + L2P.
pub fn dual_walk(
    bodies: &mut crate::bodies::Bodies,
    tree: &Octree,
    p: usize,
    eps: f64,
    pows: &[f64],
    binom: &[f64],
    fmin: &[f64],
    moments: &[C],
    locals: &mut [C],
    rot: Option<&mut RotCtx>,
) {
    let mut rot = rot;
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(1024);
    stack.push((0, 0));
    while let Some((a, b)) = stack.pop() {
        N_POP.fetch_add(1, Ordering::Relaxed);
        let ca = tree.cells[a];
        let cb = tree.cells[b];
        if a == b {
            if tree.is_leaf(a) {
                // P2P within the leaf.
                N_P2P_SELF.fetch_add(1, Ordering::Relaxed);
                crate::p2p::p2p_self(bodies, ca.first, ca.count);
            } else {
                // All pairs of daughters including (i,i).
                for i in 0..8 {
                    if ca.children[i] < 0 {
                        continue;
                    }
                    for j in i..8 {
                        if ca.children[j] >= 0 {
                            stack.push((ca.children[i] as usize, ca.children[j] as usize));
                        }
                    }
                }
            }
        } else if accept(a, b, &tree.cells, pows, binom, p, eps, fmin)
            && accept(b, a, &tree.cells, pows, binom, p, eps, fmin)
        {
            N_ACCEPT.fetch_add(1, Ordering::Relaxed);
            if p >= crate::rotation::ROT_P {
                if let Some(r) = rot.as_deref_mut() {
                    N_M2L_OPS.fetch_add(1, Ordering::Relaxed);
                    r.m2l_pair(tree, a, b, moments, locals);
                } else {
                    N_M2L_OPS.fetch_add(1, Ordering::Relaxed);
                    m2l_mutual(tree, a, b, moments, locals, p);
                }
            } else {
                N_M2L_OPS.fetch_add(1, Ordering::Relaxed);
                m2l_mutual(tree, a, b, moments, locals, p);
            }
        } else if tree.is_leaf(a) && tree.is_leaf(b) {
            N_P2P_PAIR.fetch_add(1, Ordering::Relaxed);
            crate::p2p::p2p_pair(bodies, ca.first, ca.count, cb.first, cb.count);
        } else if tree.is_leaf(a) {
            // Split the larger (non-leaf) cell b.
            for j in 0..8 {
                if cb.children[j] >= 0 {
                    stack.push((a, cb.children[j] as usize));
                }
            }
        } else if tree.is_leaf(b) {
            for i in 0..8 {
                if ca.children[i] >= 0 {
                    stack.push((ca.children[i] as usize, b));
                }
            }
        } else {
            // Split the larger cell (higher level = smaller; compare rho).
            if ca.rho >= cb.rho {
                for i in 0..8 {
                    if ca.children[i] >= 0 {
                        stack.push((ca.children[i] as usize, b));
                    }
                }
            } else {
                for j in 0..8 {
                    if cb.children[j] >= 0 {
                        stack.push((a, cb.children[j] as usize));
                    }
                }
            }
        }
    }
}