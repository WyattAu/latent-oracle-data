//! Glue between shakmaty positions and the shard record format.

use crate::shard::Record;
use shakmaty::{fen::Fen, CastlingMode, Chess, Color, EnPassantMode, FromSetup, Move, Position, Role, Square};
#[allow(unused_imports)]
use shakmaty::Setup;

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub fn startpos() -> Chess {
    Chess::default()
}

pub fn parse_san(pos: &Chess, san: &str) -> Option<Move> {
    use shakmaty::san::San;
    let parsed = San::from_ascii(san.trim().as_bytes()).ok()?;
    parsed.to_move(pos).ok()
}

pub fn apply(pos: &mut Chess, mv: &Move) {
    pos.play_unchecked(mv);
}

fn role_code(role: Role) -> u8 {
    match role {
        Role::Pawn => 1,
        Role::Knight => 2,
        Role::Bishop => 3,
        Role::Rook => 4,
        Role::Queen => 5,
        Role::King => 6,
    }
}

pub fn role_from_promo_code(code: u8) -> Option<Role> {
    Some(match code {
        0 => Role::Knight,
        1 => Role::Bishop,
        2 => Role::Rook,
        3 => Role::Queen,
        _ => return None,
    })
}

pub fn promo_code(role: Role) -> u8 {
    role_code(role) - 1
}

fn square_index(s: Square) -> u8 {
    (s as u8) % 64
}

/// Capture the position as a record (side, rights, capturable-only ep).
/// En-passant normalization comes from shakmaty's `LegalOnly` mode, which is
/// exactly the engine's convention.
pub fn capture_meta(pos: &Chess) -> (u8, u8, u8, u8, u16) {
    let board = pos.board();
    let side = match pos.turn() {
        Color::White => 0,
        Color::Black => 1,
    };
    let rights = pos.castles().castling_rights();
    // (Position::castling_rights exists on the trait; Chess delegates to Setup)
    let mut castling = 0u8;
    if rights.contains(Square::H1) {
        castling |= 1;
    }
    if rights.contains(Square::A1) {
        castling |= 2;
    }
    if rights.contains(Square::H8) {
        castling |= 4;
    }
    if rights.contains(Square::A8) {
        castling |= 8;
    }
    let fen = shakmaty::fen::Fen::from_position(pos.clone(), EnPassantMode::Legal);
    let ep = match fen.0.ep_square {
        Some(s) => square_index(s),
        None => 255,
    };
    (side, castling, ep, u8::try_from(pos.halfmoves()).unwrap_or(255), u16::try_from(u32::from(pos.fullmoves())).unwrap_or(1))
}

pub fn capture_record(pos: &Chess, target: &Move, wdl: [f32; 3]) -> Record {
    let mut r = Record::default();
    let (side, castling, ep, halfmove, fullmove) = capture_meta(pos);
    r.side = side;
    r.castling = castling;
    r.ep = ep;
    r.halfmove = halfmove;
    r.fullmove = fullmove;

    let board = pos.board();
    for i in 0u32..64 {
        let s = Square::new(i);
        if let Some(piece) = board.piece_at(s) {
            let base = match piece.color {
                Color::White => 0,
                Color::Black => 8,
            };
            r.set_piece(square_index(s) as usize, base + role_code(piece.role));
        }
    }

    r.n_targets = 1;
    r.targets[0] = (
        square_index(mv_from(target)),
        square_index(mv_to(target)),
        target
            .promotion()
            .map(promo_code)
            .unwrap_or(255),
    );
    r.wdl = wdl;
    r
}

pub fn mv_from(mv: &Move) -> Square {
    match mv {
        Move::Normal { from, .. } => *from,
        Move::EnPassant { from, .. } => *from,
        Move::Castle { king, rook: _ } => *king,
        Move::Put { .. } => unreachable!(),
    }
}

pub fn mv_to(mv: &Move) -> Square {
    match mv {
        Move::Normal { to, .. } => *to,
        Move::EnPassant { to, .. } => *to,
        Move::Castle { king, rook } => {
            // engine convention: castling is encoded king-from -> king-to
            let k = *king;
            let r = *rook;
            if (r as u8) > (k as u8) {
                // kingside: king to g-file
                Square::new((k as u32) + 2)
            } else {
                Square::new((k as u32) - 2)
            }
        }
        Move::Put { to, .. } => *to,
    }
}

/// Record -> FEN string (for Stockfish), matching the engine's conventions.
pub fn record_to_fen(r: &Record) -> String {
    let mut placement = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let code = r.piece_at((rank * 8 + file) as usize);
            let c = match code {
                1 => 'P',
                2 => 'N',
                3 => 'B',
                4 => 'R',
                5 => 'Q',
                6 => 'K',
                9 => 'p',
                10 => 'n',
                11 => 'b',
                12 => 'r',
                13 => 'q',
                14 => 'k',
                _ => {
                    empty += 1;
                    continue;
                }
            };
            if empty > 0 {
                placement.push_str(&empty.to_string());
                empty = 0;
            }
            placement.push(c);
        }
        if empty > 0 {
            placement.push_str(&empty.to_string());
        }
        if rank > 0 {
            placement.push('/');
        }
    }

    let mut castling = String::new();
    for (bit, ch) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
        if r.castling & bit != 0 {
            castling.push(ch);
        }
    }
    let castling = if castling.is_empty() {
        "-".to_string()
    } else {
        castling
    };

    let ep = if r.ep == 255 {
        "-".to_string()
    } else {
        let f = r.ep % 8;
        let rk = r.ep / 8;
        format!("{}{}", (b'a' + f) as char, (b'1' + rk) as char)
    };

    format!(
        "{} {} {} {} {} {}",
        placement,
        if r.side == 0 { "w" } else { "b" },
        castling,
        ep,
        r.halfmove,
        r.fullmove
    )
}

/// Parse a FEN into a position (tests / label spot-checks).
pub fn from_fen(fen: &str) -> Option<Chess> {
    let setup: Fen = fen.parse().ok()?;
    Chess::from_setup(setup.0, shakmaty::CastlingMode::Standard).ok()
}
