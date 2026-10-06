"""One-command verdict analysis for a fastchess match: pentanomial pair
stats + conversion metrics + game-level Elo, formatted as a RESULTS.md
table fragment. Replaces the manual per-verdict ritual."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def game_elo(txt_path: Path) -> str:
    txt = txt_path.read_text()
    # fastchess prints RUNNING SPRT estimates mid-match; the verdict is the
    # LAST Elo/Games pair in the file.
    # fastchess prints "Elo: X, nElo: Y" on ONE line, so a plain `Elo:` search
    # matches the nElo value too. \b requires a word boundary, which "nElo"
    # does not have before "Elo".
    elo = re.findall(r"\bElo: ([-0-9.]+) \+/- ([0-9.]+)", txt)
    games = re.findall(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)", txt)
    if not elo:
        return "n/a"
    e = elo[-1]
    line = f"{e[0]} ± {e[1]}"
    if games:
        g = games[-1]
        line += f" ({g[0]} games, {g[1]}W/{g[2]}L/{g[3]}D)"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_stem", help="e.g. /path/to/bcv1-final-p2 (without .pgn/.txt)")
    ap.add_argument("--net-name", default="")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    stem = Path(args.match_stem)
    pgn, txt = stem.with_suffix(".pgn"), stem.with_suffix(".txt")
    print(f"## {args.net_name or stem.name}")
    print(f"- game-level: {game_elo(txt)}")

    # pentanomial via the existing script (subprocess: it sys.exits)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "sprt_pentanomial.py"),
                        str(pgn)], capture_output=True, text=True)
    pair = [l for l in r.stdout.splitlines() if l.startswith(("pairs", "pair score", "Elo"))]
    for l in pair:
        print(f"- {l}")

    # conversion via the existing script
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "conversion_metrics.py"),
                        str(pgn)], capture_output=True, text=True)
    conv = [l for l in r.stdout.splitlines() if l.startswith(("conversion", "games", "time"))]
    for l in conv:
        print(f"- {l}")

    if args.markdown:
        print("\n```markdown")
        print(f"| {args.net_name or stem.name} | {game_elo(txt)} | "
              f"{next((l.split(': ')[1] for l in pair if l.startswith('Elo')), 'n/a')} | "
              f"{next((l.split(': ')[1].split('= ')[-1] for l in conv if l.startswith('conversion')), 'n/a')} |")
        print("```")


if __name__ == "__main__":
    main()
