"""Decoder for DeepMind ChessBench .bagz action-value files.

Each bagz file: [zstd records][i64 boundary index][i64 index_start]. Record
content: a Beam TupleCoder(StrUtf8, StrUtf8, Float) serialisation of
(fen, move, win_prob) -- length-prefixed UTF-8 strings + 4-byte LE float.
"""
import struct


def read_bagz(path):
    """Yields decompressed record bytes from a .bagz file."""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 8:
        return
    (index_start,) = struct.unpack('<Q', data[-8:])
    index_size = len(data) - index_start
    num_records = index_size // 8 - 1
    limits = data  # the C++ reader slices the WHOLE mmap with absolute offsets
    for i in range(num_records):
        end = i * 8 + index_start
        if i:
            start, stop = struct.unpack('<2q', limits[end - 8:end + 8])
        else:
            start, stop = 0, struct.unpack('<q', limits[end:end + 8])[0]
        yield data[start:stop]


def _read_varint(buf, pos):
    shift = result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _read_str(buf, pos):
    n, pos = _read_varint(buf, pos)
    s = buf[pos:pos + n].decode('utf-8')
    return s, pos + n


def decode_av_record(buf):
    """(fen, move, win_prob) from a Beam TupleCoder serialisation. Beam's
    FloatCoder is an 8-byte BIG-ENDIAN double, not a little-endian float."""
    fen, pos = _read_str(buf, 0)
    move, pos = _read_str(buf, pos)
    (win_prob,) = struct.unpack('>d', buf[pos:pos + 8])
    return fen, move, win_prob


if __name__ == '__main__':
    import sys
    n = 0
    for rec in read_bagz(sys.argv[1] if len(sys.argv) > 1 else
                         '/home/wyatt/data/chessbench/action_value_data.bag'):
        fen, move, wp = decode_av_record(rec)
        if n < 3:
            print(f"fen={fen!r}\nmove={move!r} win_prob={wp}")
        n += 1
    print(f"decoded {n} records OK")
