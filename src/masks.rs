//! `masks` subcommand: precompute legal-move index sidecars for a shard.
//!
//! Sidecar format ("LOMS", version 1):
//!   u32 magic, u16 version, u16 reserved, u64 count,
//!   u64 offsets[count + 1]  (byte offsets into payload),
//!   payload: per record — u8 n_moves, then n_moves x u16 flat indices
//!   (flat = from * 64 + to; promotions are irrelevant for the CE mask).
//!
//! Trainer side: mmap the sidecar, build the (64,64) bool mask per record
//! without python-chess. Falls back to python-chess when the sidecar is
//! absent.

use crate::chess_glue;
use crate::shard::{Record, ShardReader};
use shakmaty::{CastlingMode, Chess, FromSetup, Position};
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct MasksArgs {
    pub input: String,
    pub output: String,
}

fn record_to_position(r: &Record) -> Option<Chess> {
    let fen = chess_glue::record_to_fen(r);
    let setup: shakmaty::fen::Fen = fen.parse().ok()?;
    Chess::from_setup(setup.0, CastlingMode::Standard).ok()
}

pub fn run(args: &MasksArgs) -> Result<(), String> {
    let file = File::open(&args.input).map_err(|e| e.to_string())?;
    let mut reader: ShardReader<std::fs::File> = ShardReader::open(file).map_err(|e| e.to_string())?;

    // Build the sidecar in memory (payload ~1.6 GB + offsets ~160 MB for
    // 20M records), then write header + offsets + payload sequentially.
    let mut offsets: Vec<u64> = vec![0];
    let mut payload: Vec<u8> = Vec::with_capacity(1 << 28);
    let mut n: u64 = 0;

    while let Some(rec) = reader.next_record().map_err(|e| e.to_string())? {
        let mut moves: Vec<u16> = Vec::with_capacity(64);
        if let Some(pos_) = record_to_position(&rec) {
            for mv in pos_.legal_moves() {
                let from = u8::from(chess_glue::mv_from(&mv)) as u16;
                let to = u8::from(chess_glue::mv_to(&mv)) as u16;
                moves.push(from * 64 + to);
            }
        }
        payload.push(moves.len() as u8);
        for m in &moves {
            payload.extend_from_slice(&m.to_le_bytes());
        }
        offsets.push(payload.len() as u64);
        n += 1;
        if n.is_multiple_of(1_000_000) {
            eprintln!("masks: {n} records");
        }
    }

    let out_file = File::create(&args.output).map_err(|e| e.to_string())?;
    let mut w = BufWriter::with_capacity(1 << 20, out_file);
    w.write_all(&0x534D_4F4Cu32.to_le_bytes()).map_err(|e| e.to_string())?; // "LOMS" LE
    w.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&0u16.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
    for o in &offsets {
        w.write_all(&o.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    w.write_all(&payload).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    eprintln!("masks: done — {n} records -> {}", args.output);
    Ok(())
}
