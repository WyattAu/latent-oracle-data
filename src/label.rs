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
use std::time::Instant;

pub struct LabelArgs {
    pub input: String,
    pub output: String,
    pub sf: String,
    pub depth: u16,
    pub multipv: u8,
    pub threads: usize,
    pub max_records: u64,
    /// Skip analysis for records that already carry labels (n_targets >= 2);
    /// they are written through unchanged. Enables free 1M -> 5M continuation.
    pub resume: bool,
    pub hash_mb: u32,
    /// Records per durable batch: the granularity of crash progress. A kill
    /// costs at most this many records of work.
    pub batch_records: usize,
}

struct Patch {
    /// Record index this patch came from (provenance; also used in logs).
    #[allow(dead_code)]
    idx: u64,
    eval_cp: i16,
    targets: [(u8, u8, u8); 3],
    n_targets: u8,
}

/// (eval in centipawns from the mover's side, is_mate, pv targets)
type Analysis = (i64, bool, Vec<(u8, u8, u8)>);

struct Engine {
    /// Held so the child process lives as long as the pipes, and polled by
    /// `dead()` so a crashed engine is detected instead of hanging.
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
        e.send("setoption name Threads value 1")?;
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
            if self.dead() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "engine exited during handshake",
                ));
            }
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
    /// True if the child has already exited (zombie included).
    ///
    /// A dead engine does not always produce EOF: Stockfish can leave a forked
    /// grandchild holding the pipe's write end, so the blocking read below
    /// never returns and the whole run hangs. That happened on 2026-10-07 --
    /// six defunct children and 13 h of lost labeling. Checking for an exited
    /// child before every analysis turns that hang into a normal error, which
    /// the caller's existing path handles by respawning the engine.
    fn dead(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(Some(_)))
    }

    fn analyze(&mut self, fen: &str, depth: u16, multipv: u8) -> std::io::Result<Analysis> {
        if self.dead() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "engine already exited",
            ));
        }
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

/// Records per durable batch. Each batch is written + flushed, so a crash
/// (OOM, reboot) costs at most one batch instead of the whole run.
fn analyze_batch(args: &LabelArgs, batch: &[Record], base: u64) -> HashMap<usize, Patch> {
    let next = AtomicU64::new(0);
    let merged: Mutex<HashMap<usize, Patch>> = Mutex::new(HashMap::new());
    let failures = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..args.threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed) as usize;
                if i >= batch.len() {
                    break;
                }
                if args.resume && batch[i].n_targets >= 2 {
                    continue; // already labeled by a previous pass
                }
                ENG.with(|eng| {
                    let mut slot = eng.borrow_mut();
                    if slot.is_none() {
                        *slot = Engine::spawn(&args.sf, args.hash_mb, args.multipv).ok();
                    }
                    let Some(e) = slot.as_mut() else {
                        failures.fetch_add(1, Ordering::Relaxed);
                        return;
                    };
                    match analyze_record(e, base + i as u64, &batch[i], args.depth, args.multipv) {
                        Ok(p) => {
                            merged.lock().unwrap().insert(i, p);
                        }
                        Err(err) => {
                            // engine desynced: respawn and skip the record
                            eprintln!(
                                "label: record {} failed analysis ({err}); respawning",
                                base + i as u64
                            );
                            failures.fetch_add(1, Ordering::Relaxed);
                            *slot = None;
                        }
                    };
                });
            });
        }
    });
    if failures.load(Ordering::Relaxed) > 0 {
        eprintln!(
            "label: warning — {} records failed analysis in batch at {base}",
            failures.load(Ordering::Relaxed)
        );
    }
    merged.into_inner().unwrap()
}

pub fn run(args: &LabelArgs) -> Result<(), String> {
    let file = std::fs::File::open(&args.input).map_err(|e| e.to_string())?;
    let mut reader: ShardReader<std::fs::File> =
        ShardReader::open(file).map_err(|e| e.to_string())?;

    // Resume: a partially written output is a valid prefix of the input, so
    // the durable record count is derivable from its size.
    let out_path = std::path::Path::new(&args.output);
    let mut start_rec: u64 = 0;
    let mut writer = if out_path.exists() && args.resume {
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(out_path)
            .map_err(|e| e.to_string())?;
        let size = f.metadata().map_err(|e| e.to_string())?.len();
        let have = size.saturating_sub(crate::shard::HEADER_SIZE as u64)
            / crate::shard::RECORD_SIZE as u64;
        // drop any torn trailing record
        let good = crate::shard::HEADER_SIZE as u64 + have * crate::shard::RECORD_SIZE as u64;
        if good != size {
            eprintln!("label: trimming torn tail {size} -> {good} bytes");
        }
        f.set_len(good).map_err(|e| e.to_string())?;
        start_rec = have;
        ShardWriter::with_count(BufWriter::with_capacity(1 << 20, f), have)
            .map_err(|e| e.to_string())?
    } else {
        ShardWriter::create(BufWriter::with_capacity(
            1 << 20,
            std::fs::File::create(out_path).map_err(|e| e.to_string())?,
        ))
        .map_err(|e| e.to_string())?
    };
    if start_rec > 0 {
        eprintln!("label: resuming at record {start_rec} ({} already durable)", start_rec);
    }

    // skip the records already written
    let mut skipped = 0u64;
    while skipped < start_rec {
        match reader.next_record().map_err(|e| e.to_string())? {
            Some(_) => skipped += 1,
            None => break,
        }
    }

    let started = Instant::now();
    let mut processed: u64 = 0;
    let target = args.max_records.saturating_sub(start_rec);
    let mut batch: Vec<Record> = Vec::with_capacity(args.batch_records.max(1));

    loop {
        batch.clear();
        let batch_cap = args.batch_records.max(1);
        let want = if args.max_records > 0 {
            (args.max_records - (start_rec + processed)).min(batch_cap as u64) as usize
        } else {
            batch_cap
        };
        if want == 0 {
            break;
        }
        while batch.len() < want {
            match reader.next_record().map_err(|e| e.to_string())? {
                Some(r) => batch.push(r),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let patches = analyze_batch(args, &batch, start_rec + processed);
        for (i, r) in batch.iter().enumerate() {
            let out = match patches.get(&i) {
                Some(p) => {
                    let mut r = r.clone();
                    r.eval_cp = p.eval_cp;
                    r.targets = p.targets;
                    r.n_targets = p.n_targets;
                    r
                }
                None => r.clone(),
            };
            writer.write(&out).map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
        processed += batch.len() as u64;
        let secs = started.elapsed().as_secs().max(1);
        let rate = processed / secs;
        let eta_min = target.saturating_sub(processed) / rate.max(1) / 60;
        eprintln!(
            "label: {}/{target} done ({} durable), {} pos/s, ETA ~{} min",
            processed, writer.count(), rate, eta_min
        );
        if args.max_records > 0 && start_rec + processed >= args.max_records {
            break;
        }
    }

    let total = writer.finalize().map_err(|e| e.to_string())?;
    eprintln!("label: {} records -> {}", total, args.output);
    Ok(())
}

thread_local! {
    static ENG: std::cell::RefCell<Option<Engine>> = const { std::cell::RefCell::new(None) };
}
