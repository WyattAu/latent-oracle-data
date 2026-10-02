//! `shard` subcommand: Lichess PGN (plain or .zst) -> BC shard.

use crate::chess_glue;
use crate::pgn;
use crate::shard::{Record, ShardWriter};
use shakmaty::{Chess, Move};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

pub struct ShardArgs {
    pub input: String,
    pub out_dir: String,
    pub min_elo: i64,
    pub min_ply: usize,
    pub max_ply: usize,
    pub sample: usize,
    pub max_positions: u64,
    pub min_tc: i64,
    pub max_tc: i64,
    pub no_elo_filter: bool,
}

fn open_reader(path: &str) -> Box<dyn BufRead> {
    let file = File::open(path).expect("open input");
    if path.ends_with(".zst") {
        let dec = zstd::stream::read::Decoder::new(file).expect("zstd decoder");
        Box::new(BufReader::with_capacity(1 << 20, dec))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    }
}

fn base_time(tc: &str) -> i64 {
    // "600+5" -> 600; "-" (unlimited) -> -1
    tc.split('+')
        .next()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(-1)
}

pub fn run(args: &ShardArgs) -> Result<(), String> {
    let mut reader = open_reader(&args.input);

    std::fs::create_dir_all(&args.out_dir).map_err(|e| e.to_string())?;
    let out_path = Path::new(&args.out_dir).join("bc.shard");
    let out_file = File::create(&out_path).map_err(|e| e.to_string())?;
    let mut writer = ShardWriter::create(BufWriter::with_capacity(1 << 20, out_file))
        .map_err(|e| e.to_string())?;

    let mut seen: HashSet<u64> = HashSet::new();
    let mut games = 0u64;
    let mut kept_games = 0u64;
    let mut written = 0u64;

    loop {
        let game = pgn::next_game(&mut reader).map_err(|e| e.to_string())?;
        let Some(game) = game else { break };
        games += 1;

        if !args.no_elo_filter {
            let white_elo = game.headers.get("WhiteElo").and_then(|v| v.parse::<i64>().ok());
            let black_elo = game.headers.get("BlackElo").and_then(|v| v.parse::<i64>().ok());
            let (Some(we), Some(be)) = (white_elo, black_elo) else {
                continue;
            };
            if we < args.min_elo || be < args.min_elo {
                continue;
            }
        }
        if let Some(tc) = game.headers.get("TimeControl") {
            let bt = base_time(tc);
            if bt < args.min_tc || bt > args.max_tc {
                continue;
            }
        }
        let result = game.headers.get("Result").map(String::as_str).unwrap_or("*");
        let result_wdl = match result {
            "1-0" => [1.0, 0.0, 0.0],
            "0-1" => [0.0, 0.0, 1.0],
            "1/2-1/2" => [0.5, 0.5, 0.0],
            _ => continue,
        };
        kept_games += 1;

        let mut pos = chess_glue::startpos();

        for (ply, san) in game.moves.iter().enumerate() {
            let Some(mv) = chess_glue::parse_san(&pos, san) else {
                break; // malformed game; drop it
            };

            if ply >= args.min_ply
                && ply <= args.max_ply
                && (ply - args.min_ply) % args.sample.max(1) == 0
            {
                let rec = chess_glue::capture_record(&pos, &mv, result_wdl);
                let key = rec.key();
                if seen.insert(key) {
                    writer.write(&rec).map_err(|e| e.to_string())?;
                    written += 1;
                    if written % 1_000_000 == 0 {
                        eprintln!("shard: {written} positions ({games} games scanned)");
                    }
                }
            }

            chess_glue::apply(&mut pos, &mv);
            if written >= args.max_positions {
                break;
            }
        }

        if written >= args.max_positions {
            break;
        }
    }

    let _ = writer.finalize().map_err(|e| e.to_string())?;
    eprintln!(
        "shard: done — {written} positions from {kept_games}/{games} games; out: {}",
        out_path.display()
    );
    Ok(())
}

// Silence unused import when feature combos change.
#[allow(unused)]
fn _t(_: Move) {}
