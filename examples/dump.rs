use lo_data::shard::ShardReader;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let f = std::fs::File::open(path).unwrap();
    let mut r: ShardReader<std::fs::File> = ShardReader::open(f).unwrap();
    let mut i = 0u64;
    while let Some(rec) = r.next_record().unwrap() {
        let bad: Vec<(usize, u8)> = (0..64)
            .map(|sq| if sq % 2 == 0 { rec.board[sq/2] & 0x0F } else { rec.board[sq/2] >> 4 })
            .enumerate().filter(|(_, c)| *c > 14).collect();
        if !bad.is_empty() || i < 2 {
            println!("rec {i}: side={} castle={} ep={} board={:02x?} bad={bad:?}", i, rec.side, rec.castling, rec.board);
        }
        i += 1;
    }
}
