use lo_data::chess_glue;
use shakmaty::Position;
use lo_data::shard::ShardReader;
fn main() {
    let f = std::fs::File::open(std::env::args().nth(1).unwrap()).unwrap();
    let mut r: ShardReader<std::fs::File> = ShardReader::open(f).unwrap();
    let mut i = 0u64;
    while let Some(rec) = r.next_record().unwrap() {
        let fen = chess_glue::record_to_fen(&rec);
        let pos = chess_glue::from_fen(&fen);
        let n = pos.as_ref().map(|p| p.legal_moves().len() as u64).unwrap_or(9999);
        println!("rec{i}: fen_ok={} moves={n} fen={fen}", pos.is_some());
        i += 1;
    }
}
