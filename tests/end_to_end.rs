//! End-to-end: sample PGN -> shard -> verify contents.

use lo_data::shard::ShardReader;
use lo_data::worker::{self, ShardArgs};

const SAMPLE: &str = "\n[Event \"Test game 1\"]\n[WhiteElo \"2400\"]\n[BlackElo \"2380\"]\n[Result \"1-0\"]\n[TimeControl \"600+0\"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n\n[Event \"Test game 2\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1520\"]\n[Result \"0-1\"]\n[TimeControl \"60+0\"]\n\n1. d4 d5 2. c4 e6 0-1\n";

fn shard_args(dir: &std::path::Path) -> ShardArgs {
    ShardArgs {
        input: write_sample(),
        out_dir: dir.display().to_string(),
        min_elo: 2000,
        min_ply: 4,
        max_ply: 40,
        sample: 1,
        max_positions: 1000,
        min_tc: 120,
        max_tc: 3600,
    }
}

fn write_sample() -> String {
    let p = std::env::temp_dir().join("lo_data_test.pgn");
    std::fs::write(&p, SAMPLE).unwrap();
    p.display().to_string()
}

#[test]
fn shard_end_to_end() {
    let dir = std::env::temp_dir().join("lo_data_test_out");
    let _ = std::fs::remove_dir_all(&dir);
    let args = shard_args(&dir);
    worker::run(&args).expect("shard run");

    let f = std::fs::File::open(dir.join("bc.shard")).unwrap();
    let mut r: ShardReader<std::fs::File> = ShardReader::open(f).unwrap();
    let mut n = 0;
    let mut first: Option<lo_data::shard::Record> = None;
    let mut keys = std::collections::HashSet::new();
    while let Some(rec) = r.next_record().unwrap() {
        keys.insert(rec.key());
        if first.is_none() {
            first = Some(rec);
        }
        n += 1;
    }
    // game 1 has plies 4..=10 in range: positions at ply 4..=10 = 7 records
    // (game 2 filtered by Elo)
    assert_eq!(n, 6, "expected 6 sampled positions (plies 4..=9 of a 10-ply game)");
    assert_eq!(keys.len() as u64, n, "dedup within shard");

    let r0 = first.unwrap();
    // ply 4 = before 3.Bb5 — position after 1.e4 e5 2.Nf3 Nc6 3... wait:
    // ply index 4 = the 5th halfmove (3.Bb5), white to move
    assert_eq!(r0.side, 0);
    assert_eq!(r0.n_targets, 1);
    // target move is Bb5 = f1b5
    assert_eq!(r0.targets[0].0, crate_sq("f1"));
    assert_eq!(r0.targets[0].1, crate_sq("b5"));
    // game result 1-0
    assert_eq!(r0.wdl, [1.0, 0.0, 0.0]);
}

fn crate_sq(name: &str) -> u8 {
    let b = name.as_bytes();
    (b[1] - b'1') * 8 + (b[0] - b'a')
}
