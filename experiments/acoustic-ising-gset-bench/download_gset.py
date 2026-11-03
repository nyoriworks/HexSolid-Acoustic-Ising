"""Download standard Stanford G-set Max-Cut benchmark instances and record SHA-256."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

GSET_BASE_URL = "https://web.stanford.edu/~yyye/yyye/Gset"

# Representative G-set suite spanning N=800 -> 2,000 -> 5,000 -> 10,000 and diverse topologies/weights
INSTANCES: dict[str, dict[str, object]] = {
    "G1": {"n": 800, "m": 19176, "weights": "+1", "topology": "random_er", "bks": 11624},
    "G11": {"n": 800, "m": 1600, "weights": "+/-1", "topology": "toroidal_grid_spinglass", "bks": 564},
    "G14": {"n": 800, "m": 4694, "weights": "+1", "topology": "planar_skew", "bks": 3064},
    "G22": {"n": 2000, "m": 19990, "weights": "+1", "topology": "random_er", "bks": 13359},
    "G55": {"n": 5000, "m": 12498, "weights": "+1", "topology": "sparse_random", "bks": 10299},
    "G67": {"n": 10000, "m": 20000, "weights": "+/-1", "topology": "toroidal_grid_spinglass", "bks": 6950},
    "G70": {"n": 10000, "m": 9999, "weights": "+1", "topology": "sparse_random", "bks": 9591},
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}

    for name, meta in INSTANCES.items():
        dest = DATA_DIR / f"{name}.txt"
        url = f"{GSET_BASE_URL}/{name}"
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"Downloading {name} from {url} ...", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    dest.write_bytes(resp.read())
            except Exception:
                subprocess.run(["curl.exe", "-fsSL", url, "-o", str(dest)], check=True)
        raw = dest.read_bytes()

        lines = [ln.strip() for ln in raw.decode("ascii").splitlines() if ln.strip()]
        header = lines[0].split()
        n_hdr, m_hdr = int(header[0]), int(header[1])
        n_edges_file = len(lines) - 1
        assert n_hdr == int(meta["n"]), f"{name}: expected N={meta['n']}, got {n_hdr}"
        assert m_hdr == int(meta["m"]) == n_edges_file, f"{name}: expected M={meta['m']}, got {m_hdr}/{n_edges_file}"
        sha = hashlib.sha256(raw).hexdigest()
        manifest[name] = {
            **meta,
            "url": url,
            "sha256": sha,
            "bytes": len(raw),
        }
        print(f"  Verified {name}: N={n_hdr}, M={m_hdr}, BKS={meta['bks']}, sha256={sha[:16]}...", flush=True)

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved data/manifest.json successfully.")


if __name__ == "__main__":
    main()
