use learned_fmm::{build_graph, generate, graph_stats, normalize_to_unit_box, split_near_far, walk_coverage, ConfigTag};
use fmm::tree::Octree;

fn arg_val(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help") {
        println!("usage: graph [--config all|plummer|cube|clumps|mixed] [--n N] [--seed S] [--n-leaf L] [--p P] [--eps E]");
        return;
    }
    let config = arg_val(&args, "--config", "all");
    let n: usize = arg_val(&args, "--n", "1024").parse().unwrap();
    let seed: u64 = arg_val(&args, "--seed", "1000").parse().unwrap();
    let n_leaf: usize = arg_val(&args, "--n-leaf", "16").parse().unwrap();
    let p: usize = arg_val(&args, "--p", "8").parse().unwrap();
    let eps: f64 = arg_val(&args, "--eps", "1e-5").parse().unwrap();
    let walk = args.iter().any(|a| a == "--walk");
    let tags: Vec<ConfigTag> = if config == "all" {
        ConfigTag::all().to_vec()
    } else {
        vec![ConfigTag::parse(&config).expect("bad --config")]
    };
    for (i, &tag) in tags.iter().enumerate() {
        let mut parts = generate(tag, n, seed + i as u64);
        normalize_to_unit_box(&mut parts);
        let tree = Octree::build(&parts, n_leaf);
        if walk {
            let (g, cov) = walk_coverage(&parts, &tree, p, eps);
            let s = graph_stats(&g);
            let (_, _, fstats) = split_near_far(&parts, &tree);
            println!(
                "{} n={n} p={p} eps={eps}: cells={} edges={} mean_in={:.1} far_frac={:.3} walk_pairs={} walk_p2p={} same_level={} covered={} cross={} cross_mass={:.3}",
                tag.name(),
                s.ncells,
                s.total_edges,
                s.mean_incoming,
                fstats.far_frac,
                cov.walk_pairs,
                cov.walk_p2p,
                cov.same_level,
                cov.covered_both_dirs,
                cov.cross_level,
                cov.cross_mass_share
            );
            continue;
        }
        let g = build_graph(&parts, &tree, p, eps);
        let s = graph_stats(&g);
        println!(
            "{} n={n} p={p} eps={eps}: cells={} leaves={} edges={} max_in={} mean_in={:.1}",
            tag.name(),
            s.ncells,
            s.nleaves,
            s.total_edges,
            s.max_incoming,
            s.mean_incoming
        );
        for (l, cells, edges) in &s.per_level {
            println!("  level {l}: cells={cells} edges={edges}");
        }
    }
}
