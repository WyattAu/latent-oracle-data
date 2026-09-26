# latent-oracle-data

Data pipeline for [latent-oracle](https://github.com/WyattAu/latent-oracle):
Lichess PGN -> binary shards -> Stockfish-labeled training data.

This is a standalone **tool** repo. It never links into the engine; its output
is data (facts). License boundaries are documented in
[PROVENANCE](PROVENANCE.md).

## Shard format v1 (`.shard`)

Header: magic `LOSH` (u32 LE), version u16 = 1, u16 header size = 12, u64
record count. Then fixed **64-byte records**:

| offset | size | field |
|-------:|-----:|-------|
| 0      | 1    | side to move (0 = white, 1 = black) |
| 1      | 1    | castling rights (bit0 WK, bit1 WQ, bit2 BK, bit3 BQ) |
| 2      | 1    | en-passant target square 0-63, 255 = none (capturable-only) |
| 3      | 1    | halfmove clock |
| 4      | 2    | fullmove number (u16 LE) |
| 6      | 1    | n\_targets (1 = BC played move, up to 3 = SF multipv) |
| 7      | 1    | reserved (0) |
| 8      | 32   | board: 64 square nibbles, a1 = low nibble of byte 0; code = 0 empty, 1-6 white P N B R Q K, 9-14 black P N B R Q K |
| 40     | 2    | eval (i16 LE, centipawns, white POV; 0 for BC records) |
| 42     | 9    | targets: 3 x (from u8, to u8, promo u8; promo 0=N 1=B 2=R 3=Q, 255 = unused), SF rank order or played move first |
| 51     | 12   | wdl: 3 x f32 LE, white POV (game result for BC, else 0) |
| 63     | 1    | reserved (0) |

## Usage

```sh
# BC shards from Lichess PGN (plain or .zst)
lo-data shard --in lichess_db_standard_rated_2026-08.pgn.zst --out shards/ \
    --min-elo 2000 --min-ply 8 --max-ply 120 --max-positions 20000000

# Stockfish labels (staged run: 5M)
lo-data label --in shards/bc.shard --out shards/labeled.shard \
    --sf ./stockfish --depth 16 --multipv 3 --threads 6 --max-records 5000000

# Inspect
lo-data info shards/bc.shard
```

## Build

```sh
cargo build --release
```

Rust 1.75+.

## License

GPL-3.0-or-later (the chess-rules dependency `shakmaty` is GPL-3; this tool
never links into the Apache-2.0 engine — the license boundary and data-flow
 rationale are documented in [PROVENANCE](PROVENANCE.md)).
