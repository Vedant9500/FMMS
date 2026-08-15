# AGENTS.md — project rules for coding agents

FMMS: Rust FMM N-body simulator. Read `README.md` (motivation), `ROADMAP.md`
(gates), `PROJECT.md` (crate layout), `todo.md` (live session log / status).

## Commands

- **Build:** `cargo build` (release: `cargo build --release`)
- **Test all:** `cargo test` — 22 unit + 2 integration tests
- **Test one crate:** `cargo test -p fmm`
- **Rotation tests:** `cargo test -p fmm rotation`
- **Ignored exit gate** (N=10⁴ direct O(N²), slow): run explicitly in release
  `cargo test --release -p fmm --test exit_gate -- --ignored`
- **Scaling sweep:** `cargo run --release -p fmm-bench -- [max_n] [p] [eps]`
- **Phase profile** (timing + walk stats for one N):
  `cargo run --release -p fmm-bench -- 10000 8 1e-5 10000`
- **Lint:** `cargo clippy --all-targets` (codebase currently warning-free;
  keep it that way — a `cargo build` with zero `warning:` lines is the bar)

## Conventions

- **Math identity vs performance:** `m2l_mutual`/`m2l_pair` must stay
  bit-compatible with their naive baselines. When touching them, run the
  full test suite — 24 unit + 3 integration tests protect the algebra.
- **Rotation code** (`rotation.rs`) is the subtle one: swap-matrix row/col
  indexing (`mr`/`lc`, i32-space shifts for negative azimuths), folding
  phases via `rem_euclid` (not `%`) for negative `m`, and the
  `conj(M)·ph` imag sign. Do not "simplify" without re-verifying against
  `m2l_rot_matches_naive` and the Python checks in `/tmp/opencode`.
- **`ROT_P`** in `rotation.rs` is a *measured* crossover (26), not a
  constant to eyeball. Re-measure (4000-pair microbench) before changing.
- **MAC factor 8** in `mac.rs` is load-bearing for the rms error gate — do
  not tune it casually.
- **`C` complex type** in `harmonic.rs` has `.scale`, `.conj`, `.add`; use
  them, not raw `f64` pairs, for clarity.
- **No comments unless asked**; when adding comments keep them terse.
- **`.cargo/config.toml` is machine-specific** (`target-cpu=native`) and
  git-ignored; performance numbers assume it, portable builds omit it.
- **Profiling counters** in `walk.rs` (`walk_counts`, `AtomicU64`) are
  temporary but still used by the bench profile mode — remove only after all
  Phase-1 gates close.

## Gotchas

- Plummer positions must not collide (fract/mod based test data can produce
  identical particles → 1/r = inf; use prime multipliers).
- `RotCtx::new` asserts `p ≤ 60` (120! < f64 max).
- Release build needed for any timing; debug build is ~10× slower.
