//! Shard format v1: fixed 64-byte records (see README for the field table).

use std::io::{self, Read, Write};

pub const MAGIC: u32 = 0x4853_4F4C; // "LOSH" LE
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: u32 = 16;
pub const RECORD_SIZE: usize = 64;

#[derive(Clone)]
pub struct Record {
    pub side: u8,
    pub castling: u8,
    pub ep: u8, // 255 = none
    pub halfmove: u8,
    pub fullmove: u16,
    pub board: [u8; 32], // 64 nibbles
    pub eval_cp: i16,    // white POV
    pub targets: [(u8, u8, u8); 3],
    pub n_targets: u8,
    pub wdl: [f32; 3], // white POV
}

impl Default for Record {
    fn default() -> Self {
        Record {
            side: 0,
            castling: 0,
            ep: 255,
            halfmove: 0,
            fullmove: 1,
            board: [0; 32],
            eval_cp: 0,
            targets: [(255, 255, 255); 3],
            n_targets: 0,
            wdl: [0.0; 3],
        }
    }
}

impl Record {
    pub fn piece_at(&self, sq: usize) -> u8 {
        if sq % 2 == 0 {
            self.board[sq / 2] & 0x0F
        } else {
            self.board[sq / 2] >> 4
        }
    }

    pub fn set_piece(&mut self, sq: usize, code: u8) {
        if sq % 2 == 0 {
            self.board[sq / 2] = (self.board[sq / 2] & 0xF0) | (code & 0x0F);
        } else {
            self.board[sq / 2] = (self.board[sq / 2] & 0x0F) | (code << 4);
        }
    }

    pub fn key(&self) -> u64 {
        crate::zobrist::record_key(&self.board, self.side, self.castling, self.ep)
    }

    pub fn encode(&self) -> [u8; RECORD_SIZE] {
        let mut b = [0u8; RECORD_SIZE];
        b[0] = self.side;
        b[1] = self.castling;
        b[2] = self.ep;
        b[3] = self.halfmove;
        b[4..6].copy_from_slice(&self.fullmove.to_le_bytes());
        b[6] = self.n_targets;
        b[7] = 0;
        b[8..40].copy_from_slice(&self.board);
        b[40..42].copy_from_slice(&self.eval_cp.to_le_bytes());
        for (i, (f, t, p)) in self.targets.iter().enumerate() {
            b[42 + i * 3] = *f;
            b[43 + i * 3] = *t;
            b[44 + i * 3] = *p;
        }
        for (i, w) in self.wdl.iter().enumerate() {
            b[51 + i * 4..55 + i * 4].copy_from_slice(&w.to_le_bytes());
        }
        b[63] = 0;
        b
    }

    pub fn decode(b: &[u8; RECORD_SIZE]) -> Record {
        let mut r = Record::default();
        r.side = b[0];
        r.castling = b[1];
        r.ep = b[2];
        r.halfmove = b[3];
        r.fullmove = u16::from_le_bytes([b[4], b[5]]);
        r.n_targets = b[6];
        r.board.copy_from_slice(&b[8..40]);
        r.eval_cp = i16::from_le_bytes([b[40], b[41]]);
        for i in 0..3 {
            r.targets[i] = (b[42 + i * 3], b[43 + i * 3], b[44 + i * 3]);
        }
        for i in 0..3 {
            r.wdl[i] = f32::from_le_bytes([
                b[51 + i * 4],
                b[52 + i * 4],
                b[53 + i * 4],
                b[54 + i * 4],
            ]);
        }
        r
    }
}

pub struct ShardWriter<W: Write + io::Seek> {
    inner: W,
    count: u64,
}

impl<W: Write + io::Seek> ShardWriter<W> {
    pub fn create(mut inner: W) -> io::Result<Self> {
        inner.write_all(&MAGIC.to_le_bytes())?;
        inner.write_all(&VERSION.to_le_bytes())?;
        // u16 on the wire: HEADER_SIZE is u32 only for arithmetic comfort.
        inner.write_all(&(HEADER_SIZE as u16).to_le_bytes())?;
        inner.write_all(&0u64.to_le_bytes())?;
        Ok(ShardWriter { inner, count: 0 })
    }

    /// Open an existing shard for append: writes the header with `count`
    /// already set and positions at the end, so `finalize` patches the true
    /// total. Used by `label` to resume after a crash (2026-10-05: an OOM
    /// kill at 3.9M/4M records lost 17 h because output was written only at
    /// the end).
    pub fn with_count(mut inner: W, count: u64) -> io::Result<Self> {
        inner.write_all(&MAGIC.to_le_bytes())?;
        inner.write_all(&VERSION.to_le_bytes())?;
        inner.write_all(&(HEADER_SIZE as u16).to_le_bytes())?;
        inner.write_all(&count.to_le_bytes())?;
        inner.seek(io::SeekFrom::Start(HEADER_SIZE as u64 + count * RECORD_SIZE as u64))?;
        Ok(ShardWriter { inner, count })
    }

    pub fn write(&mut self, r: &Record) -> io::Result<()> {
        self.inner.write_all(&r.encode())?;
        self.count += 1;
        Ok(())
    }

    /// Push buffered bytes to the OS so a crash cannot lose a whole batch.
    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Seek back and patch the record count (header is fixed-size).
    pub fn finalize(mut self) -> io::Result<u64> {
        self.inner.flush()?;
        let end = self.inner.seek(io::SeekFrom::Current(0))?;
        let mut w = self.inner;
        w.seek(io::SeekFrom::Start(8))?;
        w.write_all(&self.count.to_le_bytes())?;
        w.seek(io::SeekFrom::Start(end))?;
        w.flush()?;
        Ok(self.count)
    }
}

pub struct ShardReader<R: Read> {
    inner: R,
    pub count: u64,
}

impl<R: Read> ShardReader<R> {
    pub fn open(mut inner: R) -> io::Result<Self> {
        let mut hdr = [0u8; HEADER_SIZE as usize];
        inner.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad shard magic"));
        }
        let version = u16::from_le_bytes([hdr[4], hdr[5]]);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported shard version {version}"),
            ));
        }
        let count = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        Ok(ShardReader { inner, count })
    }

    pub fn next_record(&mut self) -> io::Result<Option<Record>> {
        let mut b = [0u8; RECORD_SIZE];
        match self.inner.read_exact(&mut b) {
            Ok(()) => Ok(Some(Record::decode(&b))),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut r = Record::default();
        r.side = 1;
        r.castling = 0b1010;
        r.set_piece(0, 6); // white king a1
        r.set_piece(63, 14); // black king h8
        r.n_targets = 1;
        r.targets[0] = (12, 28, 255);
        r.eval_cp = -34;
        r.wdl = [0.5, 0.5, 0.0];
        let enc = r.encode();
        let dec = Record::decode(&enc);
        assert_eq!(dec.side, 1);
        assert_eq!(dec.piece_at(0), 6);
        assert_eq!(dec.piece_at(63), 14);
        assert_eq!(dec.piece_at(1), 0);
        assert_eq!(dec.eval_cp, -34);
        assert_eq!(dec.targets[0], (12, 28, 255));
        assert_eq!(dec.wdl[0], 0.5);
    }
}
