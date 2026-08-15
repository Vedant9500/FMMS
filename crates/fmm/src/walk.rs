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
//!
//! `dual_walk_par` parallelizes over the pair tree with a shared work stack:
//! each thread pops a batch of pairs, processes them (splitting pairs push
//! their children back onto the stack), and drains into private
//! `locals`/`Bodies` accumulators, reduced into the shared ones at the end.
//! The set of (A, B) pairs processed is identical to the serial walk; only
//! accumulation order differs, so results agree to fp reassociation rounding.

use crate::harmonic::C;
use crate::mac::accept;
use crate::rotation::RotCtx;
use crate::translate::{m2l_mutual_scratch, M2lScratch};
use crate::tree::Octree;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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

/// Per-worker walk counters (folded into the globals at thread end so the
/// parallel path never does contended atomic RMWs per pair).
#[derive(Default, Clone, Copy)]
struct Counters {
    pop: u64,
    accept: u64,
    m2l: u64,
    selfp: u64,
    pair: u64,
    m2l_ops: u64,
}

impl Counters {
    fn fold(&self) {
        N_POP.fetch_add(self.pop, Ordering::Relaxed);
        N_ACCEPT.fetch_add(self.accept, Ordering::Relaxed);
        N_M2L.fetch_add(self.m2l, Ordering::Relaxed);
        N_P2P_SELF.fetch_add(self.selfp, Ordering::Relaxed);
        N_P2P_PAIR.fetch_add(self.pair, Ordering::Relaxed);
        N_M2L_OPS.fetch_add(self.m2l_ops, Ordering::Relaxed);
    }
}

/// Handle one cell pair in the mutual walk, pushing any split daughters onto
/// `pending`. Shared by the serial `process_stack` and the parallel workers.
#[inline]
fn process_pair(
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
    a: usize,
    b: usize,
    pending: &mut Vec<(usize, usize)>,
    cnt: &mut Counters,
    scratch: &mut M2lScratch,
) {
    let mut rot = rot;
    cnt.pop += 1;
    let ca = tree.cells[a];
    let cb = tree.cells[b];
    if a == b {
        if tree.is_leaf(a) {
            // P2P within the leaf.
            cnt.selfp += 1;
            crate::p2p::p2p_self(bodies, ca.first, ca.count);
        } else {
            // All pairs of daughters including (i,i).
            for i in 0..8 {
                if ca.children[i] < 0 {
                    continue;
                }
                for j in i..8 {
                    if ca.children[j] >= 0 {
                        pending.push((ca.children[i] as usize, ca.children[j] as usize));
                    }
                }
            }
        }
    } else if accept(a, b, &tree.cells, pows, binom, p, eps, fmin)
        && accept(b, a, &tree.cells, pows, binom, p, eps, fmin)
    {
        cnt.accept += 1;
        cnt.m2l += 1;
        cnt.m2l_ops += 1;
        if p >= crate::rotation::ROT_P {
            if let Some(r) = rot.as_deref_mut() {
                r.m2l_pair(tree, a, b, moments, locals);
            } else {
                m2l_mutual_scratch(tree, a, b, moments, locals, p, scratch);
            }
        } else {
            m2l_mutual_scratch(tree, a, b, moments, locals, p, scratch);
        }
    } else if tree.is_leaf(a) && tree.is_leaf(b) {
        cnt.pair += 1;
        crate::p2p::p2p_pair(bodies, ca.first, ca.count, cb.first, cb.count);
    } else if tree.is_leaf(a) {
        // Split the larger (non-leaf) cell b.
        for j in 0..8 {
            if cb.children[j] >= 0 {
                pending.push((a, cb.children[j] as usize));
            }
        }
    } else if tree.is_leaf(b) {
        for i in 0..8 {
            if ca.children[i] >= 0 {
                pending.push((ca.children[i] as usize, b));
            }
        }
    } else {
        // Split the larger cell (higher level = smaller; compare rho).
        if ca.rho >= cb.rho {
            for i in 0..8 {
                if ca.children[i] >= 0 {
                    pending.push((ca.children[i] as usize, b));
                }
            }
        } else {
            for j in 0..8 {
                if cb.children[j] >= 0 {
                    pending.push((a, cb.children[j] as usize));
                }
            }
        }
    }
}

/// Drain a stack of cell pairs, doing the M2L/P2P work. This is the serial
/// walk body (and the reference for the parallel walk's pair decomposition).
#[inline]
fn process_stack(
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
    stack: &mut Vec<(usize, usize)>,
) {
    let mut rot = rot;
    let mut cnt = Counters::default();
    let mut scratch = M2lScratch::new(p);
    while let Some((a, b)) = stack.pop() {
        process_pair(
            bodies, tree, p, eps, pows, binom, fmin, moments, locals, rot.as_deref_mut(), a, b,
            stack, &mut cnt, &mut scratch,
        );
    }
    cnt.fold();
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
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(1024);
    stack.push((0, 0));
    process_stack(bodies, tree, p, eps, pows, binom, fmin, moments, locals, rot, &mut stack);
}

/// Pairs popped from the shared stack per lock acquisition. Small batches
/// keep a worker's M2L writes localized to the few parent cells of one
/// subtree (sibling pairs share cells), avoiding L2/DRAM write misses.
const BATCH: usize = 64;

/// Parallel dual tree walk with the same pair decomposition as `dual_walk`.
///
/// Workers share a work stack seeded with `(root, root)` and steal batches of
/// pairs; each splitting pair pushes its daughters back onto the stack, so
/// dynamic load balancing falls out of the decomposition itself (no task list
/// or serial collect phase). Each thread owns a private copy of `locals` and a
/// private `Bodies` accumulator (positions/masses cloned, accelerations
/// zeroed); results are reduced into `locals` and `bodies` afterwards.
/// Requires `nthreads * (ncells·dim + N)·16` bytes of transient memory for the
/// accumulators.
///
/// Termination: each thread owns a private stack (uncontended pop/push); a
/// thread whose stack empties steals half of a random victim's stack. `left`
/// counts pairs still in some stack or in a thread's in-flight batch, so
/// `left == 0` (observed after a batch completes) means all work is done.
pub fn dual_walk_par(
    bodies: &mut crate::bodies::Bodies,
    tree: &Octree,
    p: usize,
    eps: f64,
    pows: &[f64],
    binom: &[f64],
    fmin: &[f64],
    moments: &[C],
    locals: &mut [C],
    nthreads: usize,
) {
    let d = crate::harmonic::dim(p);
    let ncells = tree.cells.len();
    let n = bodies.x.len();
    let use_rot = p >= crate::rotation::ROT_P;

    let mut per_thread: Vec<(Vec<C>, crate::bodies::Bodies)> = (0..nthreads)
        .map(|_| {
            let mut tb = bodies.clone();
            tb.ax.iter_mut().for_each(|v| *v = 0.0);
            tb.ay.iter_mut().for_each(|v| *v = 0.0);
            tb.az.iter_mut().for_each(|v| *v = 0.0);
            tb.pot.iter_mut().for_each(|v| *v = 0.0);
            (vec![C::ZERO; ncells * d], tb)
        })
        .collect();

    let stacks: Vec<std::sync::Mutex<Vec<(usize, usize)>>> =
        (0..nthreads).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    stacks[0].lock().unwrap().push((0, 0));
    let left = AtomicUsize::new(1);
    let park = std::sync::Mutex::new(());
    let cv = std::sync::Condvar::new();

    std::thread::scope(|s| {
        let stacks = &stacks;
        let left = &left;
        let park = &park;
        let cv = &cv;
        for (i, slot) in per_thread.iter_mut().enumerate() {
            let (tl, tb) = (&mut slot.0, &mut slot.1);
            s.spawn(move || {
                let mut rot = if use_rot { Some(RotCtx::new(p)) } else { None };
                let mut batch: Vec<(usize, usize)> = Vec::with_capacity(BATCH);
                let mut pending: Vec<(usize, usize)> = Vec::with_capacity(BATCH * 8);
                let mut cnt = Counters::default();
                let mut scratch = M2lScratch::new(p);
                loop {
                    batch.clear();
                    {
                        let mut g = stacks[i].lock().unwrap();
                        let take = g.len().min(BATCH);
                        if take > 0 {
                            let total = g.len();
                            batch.extend(g.split_off(total - take));
                        }
                    }
                    if batch.is_empty() {
                        // Own stack empty: steal half of some victim's stack.
                        let mut stolen = false;
                        for v in 0..nthreads {
                            if v == i {
                                continue;
                            }
                            let mut g = stacks[v].lock().unwrap();
                            let n = g.len();
                            if n > 1 {
                                batch.extend(g.split_off(n - n / 2));
                                stolen = true;
                                break;
                            }
                        }
                        if !stolen && batch.is_empty() {
                            if left.load(Ordering::Acquire) == 0 {
                                break;
                            }
                            let mut g = park.lock().unwrap();
                            while left.load(Ordering::Acquire) > 0 {
                                g = cv.wait(g).unwrap();
                            }
                            if left.load(Ordering::Acquire) == 0 {
                                break;
                            }
                            continue;
                        }
                    }

                    pending.clear();
                    for &(a, b) in batch.iter() {
                        process_pair(
                            tb, tree, p, eps, pows, binom, fmin, moments, tl, rot.as_mut(), a, b,
                            &mut pending, &mut cnt, &mut scratch,
                        );
                    }

                    let k = pending.len();
                    let s = batch.len();
                    let mut g = stacks[i].lock().unwrap();
                    let was_empty = g.is_empty();
                    g.append(&mut pending);
                    let old = left.fetch_sub(s, Ordering::Relaxed);
                    let l = old - s + k;
                    left.fetch_add(k, Ordering::Release);
                    drop(g);
                    if was_empty || l == 0 {
                        cv.notify_all();
                    }
                }
                cnt.fold();
            });
        }
    });

    // Reduce private accumulators into the shared ones.
    for (tl, tb) in &per_thread {
        for i in 0..ncells * d {
            locals[i] = locals[i].add(tl[i]);
        }
        for k in 0..n {
            bodies.ax[k] += tb.ax[k];
            bodies.ay[k] += tb.ay[k];
            bodies.az[k] += tb.az[k];
            bodies.pot[k] += tb.pot[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::Bodies;
    use crate::harmonic::dim;
    use crate::models::{plummer, Particle};

    /// Run the walk both serial and threaded and check the M2L locals and the
    /// near-field accelerations agree to fp reassociation tolerance.
    /// `local_tol` is relative; p=8 locals are O(1) (reassociation ~1e-9),
    /// p=28 locals reach ~1e57 with cancellation (reassociation ~4e-6) but
    /// the forces from them cancel to 1e-8 (see end-to-end test).
    fn compare_walks(n: usize, p: usize, eps: f64, n_leaf: usize, nthreads: usize, local_tol: f64) {
        let parts = plummer(n, [0.0; 3], 77);
        let tree = Octree::build(&parts, n_leaf);
        let d = dim(p);
        let ncells = tree.cells.len();
        let mut moments = vec![C::ZERO; ncells * d];
        let bodies0 = Bodies::from_parts(&parts, &tree.order);
        crate::moments::compute_moments(&bodies0, &tree, p, &mut moments);
        let pows = crate::moments::multipole_power(&moments, ncells, p);
        let binom = crate::moments::binomials(p);
        let (_, fmin) = crate::mac::f_prepass(&bodies0, &tree);

        let mut bs = bodies0.clone();
        let mut ls = vec![C::ZERO; ncells * d];
        dual_walk(&mut bs, &tree, p, eps, &pows, &binom, &fmin, &moments, &mut ls, None);

        let mut bp = bodies0.clone();
        let mut lp = vec![C::ZERO; ncells * d];
        dual_walk_par(&mut bp, &tree, p, eps, &pows, &binom, &fmin, &moments, &mut lp, nthreads);

        for i in 0..ncells * d {
            // Any real pair-split bug would differ at the level of the summed
            // terms (~1 relative); `local_tol` sits well above reassociation
            // rounding while staying far below that.
            let tol = local_tol * (1.0 + ls[i].re.abs() + lp[i].re.abs() + ls[i].im.abs() + lp[i].im.abs());
            assert!(
                (ls[i].re - lp[i].re).abs() < tol && (ls[i].im - lp[i].im).abs() < tol,
                "locals[{i}]: serial ({},{}) vs par ({},{})",
                ls[i].re,
                ls[i].im,
                lp[i].re,
                lp[i].im
            );
        }
        let ntol = |a: f64, b: f64| 1e-10 * (1.0 + a.abs() + b.abs());
        for k in 0..n {
            assert!((bs.ax[k] - bp.ax[k]).abs() < ntol(bs.ax[k], bp.ax[k]), "ax[{k}]");
            assert!((bs.ay[k] - bp.ay[k]).abs() < ntol(bs.ay[k], bp.ay[k]), "ay[{k}]");
            assert!((bs.az[k] - bp.az[k]).abs() < ntol(bs.az[k], bp.az[k]), "az[{k}]");
            assert!((bs.pot[k] - bp.pot[k]).abs() < ntol(bs.pot[k], bp.pot[k]), "pot[{k}]");
        }
    }

    #[test]
    fn parallel_matches_serial_naive() {
        compare_walks(2048, 8, 1e-5, 16, 8, 1e-6);
    }

    #[test]
    fn parallel_matches_serial_rotation() {
        compare_walks(2048, 28, 1e-5, 16, 8, 1e-3);
    }

    #[test]
    fn parallel_single_thread_matches_serial() {
        compare_walks(1024, 6, 1e-4, 8, 1, 1e-6);
    }

    #[test]
    fn parallel_end_to_end_forces_match_serial() {
        // Physical output (accelerations) of threaded vs serial FMM.
        let parts = plummer(4096, [0.0; 3], 5);
        for (p, eps) in [(8, 1e-5), (28, 1e-5)] {
            let f = crate::fmm::Fmm::new(&parts, p, eps, 16);
            let sr = f.run(&parts);
            let pr = f.run_threaded(&parts, 8);
            for i in 0..parts.len() {
                for d in 0..3 {
                    let (a, b) = (sr.accel[i][d], pr.accel[i][d]);
                    let tol = 1e-8 * (1.0 + a.abs() + b.abs());
                    assert!(
                        (a - b).abs() < tol,
                        "accel[{i}][{d}] p={p}: serial {a} vs par {b}"
                    );
                }
                let (a, b) = (sr.pot[i], pr.pot[i]);
                assert!(
                    (a - b).abs() < 1e-8 * (1.0 + a.abs() + b.abs()),
                    "pot[{i}] p={p}"
                );
            }
        }
    }
}