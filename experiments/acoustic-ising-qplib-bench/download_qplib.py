"""Download the 23 Hans Mittelmann QUBO benchmark instances from QPLIB.

Instances:
  3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738, 3745, 3822,
  3832, 3838, 3850, 3852, 3877, 5721, 5725, 5755, 5875, 5881,
  5882, 5909, 5922
"""

from __future__ import annotations

import pathlib
import sys
import urllib.request
import urllib.error

QPLIB_IDS = [
    3506, 3565, 3642, 3650, 3693, 3705, 3706, 3738, 3745, 3822,
    3832, 3838, 3850, 3852, 3877, 5721, 5725, 5755, 5875, 5881,
    5882, 5909, 5922,
]

DATA_DIR = pathlib.Path(__file__).parent / "data"

def download_file(url: str, dest_path: pathlib.Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            dest_path.write_bytes(content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(QPLIB_IDS)} QPLIB instances into {DATA_DIR}...")
    success_count = 0
    for qid in QPLIB_IDS:
        qplib_url = f"https://qplib.zib.de/qplib/QPLIB_{qid}.qplib"
        sol_url = f"https://qplib.zib.de/sol/QPLIB_{qid}.sol"
        
        qplib_path = DATA_DIR / f"QPLIB_{qid}.qplib"
        sol_path = DATA_DIR / f"QPLIB_{qid}.sol"
        
        ok_q = download_file(qplib_url, qplib_path)
        ok_s = download_file(sol_url, sol_path)
        
        if ok_q:
            size_kb = qplib_path.stat().st_size / 1024.0
            print(f"  [QPLIB_{qid}] .qplib: {size_kb:.1f} KB, sol: {'OK' if ok_s else 'None'}")
            success_count += 1
        else:
            print(f"  [QPLIB_{qid}] FAILED to download .qplib")
            
    print(f"Completed: {success_count}/{len(QPLIB_IDS)} instances downloaded.")
    return 0 if success_count == len(QPLIB_IDS) else 1

if __name__ == "__main__":
    sys.exit(main())

