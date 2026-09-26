//! `label` subcommand: patch Stockfish evaluations + multipv PV moves into a
//! shard copy. Thread pool of SF processes; scoped threads borrow the record
//! buffer and each returns its patches.

use crate::chess_glue::record_to_fen;
use crate::shard::{Record, ShardReader, ShardWriter};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub struct LabelArgs {
    pub input: String,
    pub output: String,
    pub sf: String,
    pub depth: u16,
    pub multipv: u8,
    pub threads: usize,
    pub max_records: u64,
    pub hash_mb: u32,
}

struct Patch {
    idx: u64,
    eval_cp: i16,
    targets: [(u8, u8, u8); 3],
    n_targets: u8,
}

struct Engine {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl Engine {
    fn spawn(path: &str, hash_mb: u32, multipv: u8) -> std::io::Result<Engine> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let mut e = Engine {
            child,
            stdin,
            reader: BufReader::with_capacity(1 << 16, stdout),
        };
        e.send("uci")?;
        e.wait_for("uciok")?;
        e.send(&format!("setoption name Threads value 1"))?;
        e.send(&format!("setoption name Hash value {hash_mb}"))?;
        e.send(&format!("setoption name Multipv value {multipv}"))?;
        e.send("isready")?;
        e.wait_for("readyok")?;
        Ok(e)
    }

    fn send(&mut self, cmd: &str) -> std::io::Result<()> {
        self.stdin.write_all(cmd.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()
    }

    fn wait_for(&mut self, token: &str) -> std::io::Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "engine died",
                ));
            }
            if line.contains(token) {
                return Ok(());
            }
        }
    }

    /// Analyze the mover-POV position; returns (mover_cp, mate?, pv targets).
    fn analyze(&mut self, fen: &str, depth: u16, multipv: u8) -> std::io::Result<(i64, bool, Vec<(u8, u8, u8)>)> {
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go depth {depth}"))?;

        let mut score: (i64, bool) = (0, false);
        let mut score_seen = false;
        let mut pv: Vec<Option<(u8, u8, u8)>> = vec![None; multipv as usize];
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "engine died",
                ));
            }
            if line.starts_with("bestmove") {
                break;
            }
            if !line.contains(" multipv ") || !line.contains(" pv ") {
                continue;
            }
            let mpv: usize = line
                .split(" multipv ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if mpv == 0 || mpv > multipv as usize {
                continue;
            }
            if let Some(after) = line.split(" score ").nth(1) {
                let mut it = after.split_whitespace();
                let kind = it.next().unwrap_or("");
                let val: i64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                match kind {
                    "cp" => {
                        score = (val, false);
                        score_seen = true;
                    }
                    "mate" => {
                        // signed mate distance; map to ±(15000 - plies)
                        let v = val.signum() * (15000 - val.unsigned_abs().min(15000)) as i64;
                        score = (v, true);
                        score_seen = true;
                    }
                    _ => {}
                }
            }
            if let Some(pvm) = line.split(" pv ").nth(1).and_then(|s| s.split_whitespace().next()) {
                pv[mpv - 1] = parse_uci_move(pvm);
            }
        }
        if !score_seen {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no score in analysis",
            ));
        }
        Ok((score.0, score.1, pv.into_iter().flatten().collect()))
    }
}

fn sq_index(file_char: u8, rank_char: u8) -> Option<u8> {
    let f = file_char.checked_sub(b'a')?;
    let r = rank_char.checked_sub(b'1')?;
    if f < 8 && r < 8 {
        Some(r * 8 + f)
    } else {
        None
    }
}

fn parse_uci_move(s: &str) -> Option<(u8, u8, u8)> {
    let b = s.as_bytes();
    if b.len() < 4 {
        return None;
    }
    let from = sq_index(b[0], b[1])?;
    let to = sq_index(b[2], b[3])?;
    let promo = if b.len() >= 5 {
        match b[4] {
            b'n' => 0,
            b'b' => 1,
            b'r' => 2,
            b'q' => 3,
            _ => return None,
        }
    } else {
        255
    };
    Some((from, to, promo))
}

fn analyze_record(
    e: &mut Engine,
    idx: u64,
    r: &Record,
    depth: u16,
    multipv: u8,
) -> std::io::Result<Patch> {
    let fen = record_to_fen(r);
    let (mover_cp, _mate, pv) = e.analyze(&fen, depth, multipv)?;
    // SF analyzes mover-POV; shard evals are white-POV.
    let white_cp = if r.side == 1 { -mover_cp } else { mover_cp };
    let mut targets = [(255u8, 255u8, 255u8); 3];
    let mut n = 0u8;
    for t in pv.into_iter().take(3) {
        targets[n as usize] = t;
        n += 1;
    }
    Ok(Patch {
        idx,
        eval_cp: white_cp.clamp(-15000, 15000) as i16,
        targets,
        n_targets: n,
    })
}

pub fn run(args: &LabelArgs) -> Result<(), String> {
    let file = std::fs::File::open(&args.input).map_err(|e| e.to_string())?;
    let mut reader: ShardReader<std::fs::File> = ShardReader::open(file).map_err(|e| e.to_string())?;
    let mut records: Vec<Record> = Vec::new();
    while records.len() < args.max_records as usize {
        match reader.next_record().map_err(|e| e.to_string())? {
            Some(r) => records.push(r),
            None => break,
        }
    }
    eprintln!("label: {} records loaded", records.len());

    let next = AtomicU64::new(0);
    let merged: Mutex<HashMap<u64, Patch>> = Mutex::new(HashMap::new());
    let failures = AtomicU64::new(0);

    std::thread::scope(|scope| {
        for _ in 0..args.threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i as usize >= records.len() {
                    break;
                }
                // Thread-local engine is created lazily inside the closure
                // via thread_local! below.
                ENG.with(|eng| {
                    let mut slot = eng.borrow_mut();
                    if slot.is_none() {
                        *slot = Engine::spawn(&args.sf, args.hash_mb, args.multipv).ok();
                    }
                    let Some(e) = slot.as_mut() else {
                        failures.fetch_add(1, Ordering::Relaxed);
                        return;
                    };
                    match analyze_record(e, i, &records[i as usize], args.depth, args.multipv) {
                        Ok(p) => merged.lock().unwrap().insert(i, p),
                        Err(_) => {
                            // engine desynced: respawn and skip the record
                            failures.fetch_add(1, Ordering::Relaxed);
                            *slot = None;
                            None
                        }
                    };
                });
            });
        }
    });

    let failures = failures.load(Ordering::Relaxed);
    if failures > 0 {
        eprintln!("label: warning — {failures} records failed analysis");
    }

    let out_file = std::fs::File::create(&args.output).map_err(|e| e.to_string())?;
    let mut writer =
        ShardWriter::create(BufWriter::with_capacity(1 << 20, out_file)).map_err(|e| e.to_string())?;
    let patches = merged.into_inner().unwrap();
    for (i, r) in records.iter().enumerate() {
        let r = match patches.get(&(i as u64)) {
            Some(p) => {
                let mut r = r.clone();
                r.eval_cp = p.eval_cp;
                r.targets = p.targets;
                r.n_targets = p.n_targets;
                r
            }
            None => r.clone(),
        };
        writer.write(&r).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    eprintln!("label: {} patched -> {}", patches.len(), args.output);
    Ok(())
}

thread_local! {
    static ENG: std::cell::RefCell<Option<Engine>> = const { std::cell::RefCell::new(None) };
}
