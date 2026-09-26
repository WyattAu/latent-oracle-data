use lo_data::shard::ShardReader;
fn main() {
    let f = std::fs::File::open(std::env::args().nth(1).unwrap()).unwrap();
    let mut r: ShardReader<std::fs::File> = ShardReader::open(f).unwrap();
    let mut i = 0;
    while let Some(rec) = r.next_record().unwrap() {
        println!("rec{i}: eval={} n={} t0=({},{},{}) wdl=({:.1},{:.1},{:.1})",
            rec.eval_cp, rec.n_targets, rec.targets[0].0, rec.targets[0].1, rec.targets[0].2,
            rec.wdl[0], rec.wdl[1], rec.wdl[2]);
        i += 1;
        if i >= 3 { break; }
    }
}
