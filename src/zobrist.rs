//! Zobrist keys, bit-identical to the engine's `src/types.hpp`
//! (same splitmix64 stream, same seed, same layout) so dedup keys and
//! cross-references between shard and engine agree exactly.

use std::sync::OnceLock;

pub struct ZobristKeys {
    pub piece: [[[u64; 64]; 6]; 2], // [color][piece_type][square], a1 = 0
    pub castle: [u64; 16],          // bit0 WK, bit1 WQ, bit2 BK, bit3 BQ
    pub ep: [u64; 8],
    pub side: u64,
}

fn splitmix_next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *s;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn make_keys() -> ZobristKeys {
    let mut s: u64 = 0x0123_4567_89AB_CDE7;
    let mut z = ZobristKeys {
        piece: [[[0; 64]; 6]; 2],
        castle: [0; 16],
        ep: [0; 8],
        side: 0,
    };
    for c in 0..2 {
        for p in 0..6 {
            for sq in 0..64 {
                z.piece[c][p][sq] = splitmix_next(&mut s);
            }
        }
    }
    for i in 0..16 {
        z.castle[i] = splitmix_next(&mut s);
    }
    for i in 0..8 {
        z.ep[i] = splitmix_next(&mut s);
    }
    z.side = splitmix_next(&mut s);
    z
}

pub fn keys() -> &'static ZobristKeys {
    static KEYS: OnceLock<ZobristKeys> = OnceLock::new();
    KEYS.get_or_init(make_keys)
}

/// zobrist key for a raw record (same field semantics as the C++ engine).
pub fn record_key(
    board: &[u8; 32], // 64 nibbles, code = (black?8:0) + ptype(1..=6), 0 empty
    side: u8,
    castling: u8,
    ep: u8, // 0-63 or 255
) -> u64 {
    let k = keys();
    let mut key: u64 = 0;
    for sq in 0..64usize {
        let code = if sq % 2 == 0 {
            board[sq / 2] & 0x0F
        } else {
            board[sq / 2] >> 4
        };
        if code == 0 || code > 14 {
            continue; // empty or corrupt nibble: skip rather than panic
        }
        let black = ((code >> 3) & 1) as usize;
        let ptype = ((code & 7) - 1) as usize; // 0..=5
        key ^= k.piece[black][ptype][sq];
    }
    key ^= k.castle[castling as usize];
    if ep != 255 {
        key ^= k.ep[(ep & 7) as usize];
    }
    if side == 1 {
        key ^= k.side;
    }
    key
}
