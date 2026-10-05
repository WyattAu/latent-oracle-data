use lo_data::{label, masks, openings, worker};
use std::collections::HashMap;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        usage();
    }
    let cmd = argv[1].as_str();
    let kv: HashMap<String, String> = parse_kv(&argv[2..]);

    let result = match cmd {
        "shard" => worker::run(&worker::ShardArgs {
            input: req(&kv, "--in"),
            out_dir: kv.get("--out").cloned().unwrap_or_else(|| "shards".into()),
            min_elo: opt(&kv, "--min-elo", 2000),
            min_ply: opt(&kv, "--min-ply", 8) as usize,
            max_ply: opt(&kv, "--max-ply", 120) as usize,
            sample: opt(&kv, "--sample", 1) as usize,
            max_positions: opt(&kv, "--max-positions", 20_000_000),
            min_tc: opt(&kv, "--min-tc", 120),
            max_tc: opt(&kv, "--max-tc", 3600),
            no_elo_filter: kv.contains_key("--no-elo-filter"),
        }),
        "label" => label::run(&label::LabelArgs {
            input: req(&kv, "--in"),
            output: req(&kv, "--out"),
            sf: kv.get("--sf").cloned().unwrap_or_else(|| "stockfish".into()),
            depth: opt(&kv, "--depth", 16) as u16,
            multipv: opt(&kv, "--multipv", 3) as u8,
            threads: opt(&kv, "--threads", 6) as usize,
            max_records: opt(&kv, "--max-records", 5_000_000),
            resume: kv.contains_key("--resume"),
            hash_mb: opt(&kv, "--hash", 64) as u32,
            batch_records: opt(&kv, "--batch-records", 50_000) as usize,
        }),
        "masks" => masks::run(&masks::MasksArgs {
            input: req(&kv, "--in"),
            output: req(&kv, "--out"),
        }),
        "openings" => openings::run(&openings::OpeningsArgs {
            input: req(&kv, "--in"),
            output: req(&kv, "--out"),
            count: opt(&kv, "--count", 2000),
        }),
        "info" => {
            let path = req(&kv, "--in");
            info(&path)
        }
        _ => {
            usage();
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn info(path: &str) -> Result<(), String> {
    use lo_data::shard::ShardReader;
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut r = ShardReader::<std::fs::File>::open(f).map_err(|e| e.to_string())?;
    let mut n = 0u64;
    let mut labeled = 0u64;
    let mut t1 = 0u64;
    while let Some(rec) = r.next_record().map_err(|e| e.to_string())? {
        n += 1;
        if rec.n_targets > 0 {
            labeled += 1;
        }
        if rec.n_targets >= 2 {
            t1 += 1;
        }
    }
    println!("{path}: {n} records ({labeled} labeled, {t1} multi-target)");
    Ok(())
}

fn parse_kv(argv: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < argv.len() {
        if argv[i].starts_with("--") {
            if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                m.insert(argv[i].clone(), argv[i + 1].clone());
                i += 2;
            } else {
                m.insert(argv[i].clone(), String::new());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    m
}

fn req(kv: &HashMap<String, String>, key: &str) -> String {
    kv.get(key).cloned().unwrap_or_else(|| {
        eprintln!("error: missing required option {key}");
        std::process::exit(2);
    })
}

fn opt<T: std::str::FromStr>(kv: &HashMap<String, String>, key: &str, default: T) -> T {
    kv.get(key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  lo-data shard --in <file.pgn[.zst]> [--out dir] [--min-elo 2000] [--min-ply 8] \
         [--max-ply 120] [--sample 1] [--max-positions N] [--min-tc 120] [--max-tc 3600]\n  \
         lo-data label --in <in.shard> --out <out.shard> [--sf stockfish] [--depth 16] \
         [--multipv 3] [--threads 6] [--max-records N] [--hash 64]\n  \
         lo-data masks --in <file.shard> --out <file.mask>\n  \
         lo-data openings --in <file.shard> --out <file.epd> [--count 2000]\n  \
         lo-data info --in <file.shard>"
    );
    std::process::exit(2);
}
