# Provenance & License Boundaries

Engine repo: https://github.com/WyattAu/latent-oracle (Apache-2.0, clean-room).

This data-pipeline repo is a **standalone tool**. It never links into the
engine binary. Its outputs are data records (facts), which carry no license
obligations from the tools that produced them.

## Dependencies

| Crate | License | Role |
|-------|---------|------|
| shakmaty | GPL-3.0-or-later | chess rules, SAN parsing, PGN replay (this makes the *tool* GPL; the engine repo is unaffected — no code or linking crosses the boundary) |
| zstd | MIT/Apache-2.0 | streaming decompression of Lichess `.pgn.zst` |

No GPL code is read, copied, or linked. Stockfish is executed as an external
binary by the `label` subcommand to generate evaluations and PV moves; labels
are facts (this is the same boundary as the engine repo's clean-room
protocol, and standard practice across the field).

## Zobrist compatibility

`src/zobrist.rs` mirrors the deterministic splitmix64 tables of the engine's
`src/types.hpp` (same seed, same layout) so dedup keys and any future
cross-referencing between shard and engine agree bit-for-bit. Verified
against the engine by test vectors in both repos.

## Data source

Lichess open game database (https://database.lichess.org), CC0.
