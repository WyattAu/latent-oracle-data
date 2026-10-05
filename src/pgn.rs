//! Streaming PGN reader: yields (headers, SAN token list) per game.
//! Handles .pgn line structure: header block, blank line, movetext
//! (comments `{}`, NAGs `$n`, nested variations `()`, move numbers),
//! terminated by a result token.

use std::collections::HashMap;
use std::io::{self, BufRead};

pub struct Game {
    pub headers: HashMap<String, String>,
    pub moves: Vec<String>,
}

const RESULTS: [&str; 4] = ["1-0", "0-1", "1/2-1/2", "*"];

fn is_result(tok: &str) -> bool {
    RESULTS.contains(&tok)
}

/// Strip {}, (), $n inline; returns clean SAN-ish token stream for one line.
fn strip_movetext(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut brace = 0usize;
    let mut paren = 0usize;
    let mut iter = line.chars().peekable();
    while let Some(c) = iter.next() {
        match c {
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            ';' => break, // rest-of-line comment
            '$' => {
                // skip NAG digits
                while let Some(&n) = iter.peek() {
                    if n.is_ascii_digit() {
                        iter.next();
                    } else {
                        break;
                    }
                }
            }
            _c if brace > 0 || paren > 0 => {}
            c => out.push(c),
        }
    }
    out
}

pub fn next_game<R: BufRead>(reader: &mut R) -> io::Result<Option<Game>> {
    let mut headers: HashMap<String, String> = HashMap::new();
    let mut moves = String::new();

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break; // EOF
        }
        let trimmed = line.trim_end();
        let t = trimmed.trim();

        if t.is_empty() {
            if !headers.is_empty() && !moves.is_empty() {
                break; // game complete
            }
            continue;
        }

        if t.starts_with('[') && moves.is_empty() {
            // header line: [Key "Value"]
            if let (Some(k0), Some(v0)) = (t.find('"'), t.rfind('"')) {
                if k0 < v0 {
                    let key = t[1..k0].trim().to_string();
                    let val = t[k0 + 1..v0].to_string();
                    headers.insert(key, val);
                }
            }
            continue;
        }

        // movetext
        moves.push(' ');
        moves.push_str(&strip_movetext(t));
    }

    if headers.is_empty() && moves.is_empty() {
        return Ok(None);
    }

    // Tokenize move text.
    let tokens: Vec<String> = moves
        .split_whitespace()
        .filter(|t| !is_result(t))
        .map(|t| {
            // Strip leading move number pattern like "1." or "12..." — the Lc0 PGN
            // format fuses the move number with the first move of each pair
            // (e.g. "1.e4" instead of "1. e4"), so this must be stripped per-token.
            let stripped = t.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
            stripped.to_string()
        })
        .filter(|t| !is_result(t) && !t.starts_with('-'))
        .filter(|t| {
            if is_result(t) {
                return false;
            }
            // drop move numbers like "1." "12..." and stray dots
            let cleaned = t.trim_end_matches('.');
            !cleaned.is_empty() && !cleaned.chars().all(|c| c.is_ascii_digit())
                || (cleaned.len() > t.len() && !cleaned.is_empty())
        })
        .filter(|t| !{
            let c = t.trim_end_matches('.');
            !c.is_empty() && c.chars().all(|c| c.is_ascii_digit())
        })
        .map(|s| s.to_string())
        .collect();

    let game = Game {
        headers,
        moves: tokens,
    };
    Ok(Some(game))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "[Event \"Test\"]\n[WhiteElo \"2400\"]\n[BlackElo \"2350\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 {best} a6 4. Ba4 Nf6 1-0\n\n[Event \"Two\"]\n[WhiteElo \"2100\"]\n[BlackElo \"2150\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n";

    #[test]
    fn reads_two_games() {
        let mut reader = SAMPLE.as_bytes();
        let g1 = next_game(&mut reader).unwrap().unwrap();
        assert_eq!(g1.headers["Result"], "1-0");
        assert_eq!(g1.moves, vec!["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]);
        let g2 = next_game(&mut reader).unwrap().unwrap();
        assert_eq!(g2.headers["Result"], "0-1");
        assert_eq!(g2.moves, vec!["d4", "d5"]);
        assert!(next_game(&mut reader).unwrap().is_none());
    }
}
