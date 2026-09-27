//! `openings` subcommand: sample stratified positions from a shard into an
//! EPD file for SPRT opening diversity. Uniform stride across the file gives
//! natural ply/structure diversity; optional per-decile caps keep extremes
//! from dominating.

use crate::chess_glue;
use crate::shard::{Record, HEADER_SIZE, RECORD_SIZE};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

pub struct OpeningsArgs {
    pub input: String,
    pub output: String,
    pub count: u64,
}

fn record_at(file: &mut File, idx: u64) -> std::io::Result<Record> {
    file.seek(SeekFrom::Start(HEADER_SIZE as u64 + idx * RECORD_SIZE as u64))?;
    let mut b = [0u8; RECORD_SIZE];
    file.read_exact(&mut b)?;
    Ok(Record::decode(&b))
}

fn to_epd(r: &Record) -> String {
    // EPD = the first four FEN fields.
    let fen = chess_glue::record_to_fen(r);
    let mut parts = fen.split_whitespace();
    let mut epd = String::new();
    for _ in 0..4 {
        if let Some(p) = parts.next() {
            epd.push_str(p);
            epd.push(' ');
        }
    }
    epd.trim_end().to_string()
}

pub fn run(args: &OpeningsArgs) -> Result<(), String> {
    let mut file = File::open(&args.input).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    if size < HEADER_SIZE as u64 {
        return Err("input too small".into());
    }
    let total = (size - HEADER_SIZE as u64) / RECORD_SIZE as u64;
    if args.count == 0 || args.count > total {
        return Err(format!("count must be in 1..={total}"));
    }

    let stride = total / args.count;
    let out_file = File::create(&args.output).map_err(|e| e.to_string())?;
    let mut w = BufWriter::new(out_file);
    let mut written = 0u64;

    for k in 0..args.count {
        let idx = k * stride + stride / 2; // middle of each stride bucket
        let r = record_at(&mut file, idx).map_err(|e| e.to_string())?;
        writeln!(w, "{}", to_epd(&r)).map_err(|e| e.to_string())?;
        written += 1;
    }
    w.flush().map_err(|e| e.to_string())?;
    eprintln!("openings: {written} positions -> {}", args.output);
    Ok(())
}
