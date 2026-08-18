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
//! `dual_walk_par` runs in two phases with the same pair decomposition: a
//! work-stealing walk that records accepted M2L pairs (near-field P2P is done
//! inline into per-thread `Bodies` accumulators), then a bucketed replay of
//! the recorded pairs into per-thread `locals` accumulators, reduced into the
//! shared ones at the end. The set of (A, B) pairs processed is identical to
//! the serial walk; only accumulation order differs, so results agree to fp
//! reassociation rounding.

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
/// With `record = Some`, accepted M2L pairs are appended to the recorder
/// instead of being evaluated (used by `dual_walk_par` phase 1; phase 2
/// replays them). The evaluated pair set is identical either way.
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
    record: Option<&mut Vec<(u32, u32)>>,
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
        if let Some(rec) = record {
            rec.push((a as u32, b as u32));
        } else if p >= crate::rotation::ROT_P {
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
            bodies, tree, p, eps, pows, binom, fmin, moments, locals, rot.as_deref_mut(), None,
            a, b, stack, &mut cnt, &mut scratch,
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

/// In-place partition of recorded pairs into contiguous buckets (American-
/// flag MSD sort, one level) keyed by `key`. DFS cell ids make each bucket a
/// subtree range, so replaying bucket-by-bucket keeps the keyed cell's
/// moments/locals cache-resident. Returns bucket end offsets (length
/// `nbuckets + 1`).
fn partition_pairs_by(
    pairs: &mut [(u32, u32)],
    ncells: usize,
    nbuckets: usize,
    key: impl Fn((u32, u32)) -> u32,
) -> Vec<usize> {
    let bucket = |id: u32| -> usize { (id as usize * nbuckets / ncells.max(1)).min(nbuckets - 1) };
    let mut ends = vec![0usize; nbuckets + 1];
    for &pr in pairs.iter() {
        ends[bucket(key(pr)) + 1] += 1;
    }
    for i in 0..nbuckets {
        ends[i + 1] += ends[i];
    }
    let mut next = ends.clone();
    for i in 0..nbuckets {
        while next[i] < ends[i + 1] {
            let pr = pairs[next[i]];
            let j = bucket(key(pr));
            if j == i {
                next[i] += 1;
            } else {
                let d = next[j];
                next[j] += 1;
                pairs.swap(next[i], d);
            }
        }
    }
    ends
}

/// Partition by `min(a, b)` — each mutual pair sits in its earlier cell's
/// subtree bucket.
fn partition_pairs(pairs: &mut [(u32, u32)], ncells: usize, nbuckets: usize) -> Vec<usize> {
    partition_pairs_by(pairs, ncells, nbuckets, |(a, b)| a.min(b))
}

/// Parallel dual tree walk with the same pair decomposition as `dual_walk`.
///
/// Two phases, to keep M2L accumulation cache-local:
///
/// 1. Work-stealing dual walk (per-thread stacks, half-stealing, `left`
///    counter for termination) that only *records* accepted M2L pairs;
///    near-field P2P accumulates directly into per-thread `Bodies` copies.
/// 2. The recorded pairs are partitioned in place into `min(a,b)` buckets
///    (subtree ranges) and replayed — mutual kernel (or the rotation kernel
///    for p ≥ ROT_P) — into per-thread `locals` accumulators, contiguous
///    bucket ranges per thread, then reduced.
///
/// A one-directional "dst-slab" replay (shared locals, no per-thread copies,
/// see `m2l_one_sided_scratch`) was prototyped and measured 2026-08-16: it
/// lost ~1.9× on kernel cost — the mutual kernel computes both directions
/// per inner-loop iteration sharing loads and one Θ table, while two
/// one-sided ops pay table build + loop overhead twice — and the memory
/// saving did not compensate at p=8. Revisit only with a cheaper one-sided
/// kernel (stage 3).
///
/// The set of (A, B) M2L pairs is identical to the serial walk; only
/// accumulation order differs, so results agree to fp reassociation rounding.
/// Requires `nthreads * (ncells·dim + N)·16` bytes of transient memory for
/// the accumulators plus `8` bytes per recorded M2L pair.
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

    // Phase 1: walk + record; P2P into private bodies copies.
    let mut workers: Vec<(Vec<(u32, u32)>, crate::bodies::Bodies)> = (0..nthreads)
        .map(|_| {
            let mut tb = bodies.clone();
            tb.ax.iter_mut().for_each(|v| *v = 0.0);
            tb.ay.iter_mut().for_each(|v| *v = 0.0);
            tb.az.iter_mut().for_each(|v| *v = 0.0);
            tb.pot.iter_mut().for_each(|v| *v = 0.0);
            (Vec::new(), tb)
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
        for (i, slot) in workers.iter_mut().enumerate() {
            let (rec, tb) = (&mut slot.0, &mut slot.1);
            s.spawn(move || {
                let mut batch: Vec<(usize, usize)> = Vec::with_capacity(BATCH);
                let mut pending: Vec<(usize, usize)> = Vec::with_capacity(BATCH * 8);
                let mut cnt = Counters::default();
                let mut scratch = M2lScratch::new(p);
                let mut nolocals: [C; 0] = [];
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
                            tb, tree, p, eps, pows, binom, fmin, moments, &mut nolocals, None,
                            Some(rec), a, b, &mut pending, &mut cnt, &mut scratch,
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

    // Merge recorded pairs and partition by min-cell subtree bucket.
    let npairs: usize = workers.iter().map(|(r, _)| r.len()).sum();
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(npairs);
    for (r, _) in &mut workers {
        pairs.append(r);
    }
    let nbuckets = 1024.min(ncells).max(1);
    let ends = partition_pairs(&mut pairs, ncells, nbuckets);

    // Phase 2: replay into per-thread locals, contiguous bucket ranges.
    let mut tls: Vec<Vec<C>> = (0..nthreads).map(|_| vec![C::ZERO; ncells * d]).collect();
    let target = npairs.div_ceil(nthreads);
    let mut bounds: Vec<usize> = vec![0];
    let mut acc = 0usize;
    for i in 0..nbuckets {
        acc += ends[i + 1] - ends[i];
        if acc >= target && bounds.len() < nthreads {
            acc = 0;
            bounds.push(ends[i + 1]);
        }
    }
    std::thread::scope(|s| {
        let mut it = tls.iter_mut();
        let pairs = &pairs;
        let nb = bounds.len();
        for j in 0..nb {
            let lo = bounds[j];
            let hi = if j + 1 < nb { bounds[j + 1] } else { npairs };
            let tl = it.next().unwrap();
            if hi <= lo {
                continue;
            }
            let mut rot = if use_rot { Some(RotCtx::new(p)) } else { None };
            let mut scratch = M2lScratch::new(p);
            s.spawn(move || {
                for &(a, b) in &pairs[lo..hi] {
                    match rot.as_mut() {
                        Some(r) => r.m2l_pair(tree, a as usize, b as usize, moments, tl),
                        None => m2l_mutual_scratch(
                            tree, a as usize, b as usize, moments, tl, p, &mut scratch,
                        ),
                    }
                }
            });
        }
    });

    // Reduce private accumulators into the shared ones.
    for (tl, (_, tb)) in tls.iter().zip(&workers) {
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

    #[test]
    fn partition_pairs_buckets_and_preserves_multiset() {
        let ncells = 1000;
        let nbuckets = 16;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut rng = 12345u64;
        let mut next = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as u32 % ncells as u32
        };
        for _ in 0..5000 {
            pairs.push((next(), next()));
        }
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        let ends = partition_pairs(&mut pairs, ncells, nbuckets);
        assert_eq!(ends.len(), nbuckets + 1);
        assert_eq!(*ends.last().unwrap(), pairs.len());
        for i in 0..nbuckets {
            assert!(ends[i] <= ends[i + 1]);
            for &(a, b) in &pairs[ends[i]..ends[i + 1]] {
                let expect = (a.min(b) as usize * nbuckets / ncells).min(nbuckets - 1);
                assert_eq!(expect, i, "pair ({a},{b}) in bucket {i}");
            }
        }
        pairs.sort_unstable();
        assert_eq!(pairs, sorted);
    }

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