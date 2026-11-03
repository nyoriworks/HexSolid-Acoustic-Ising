"""Format G-set comparison report for Duffing-Parametron (Virtual & Real Audio B=1 and B=8)."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

THIS_DIR = Path(__file__).resolve().parent
DUFFING_PATH = THIS_DIR / "results_duffing" / "summary_duffing.json"
PREV_SUMMARY = THIS_DIR / "results" / "summary.json"
MANIFEST_PATH = THIS_DIR / "data" / "manifest.json"

INST_ORDER = ("G1", "G11", "G14", "G22", "G55", "G67", "G70")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    recs = json.loads(DUFFING_PATH.read_text())
    by_key = {(r["instance"], r["solver"], r["batch_size"]): r for r in recs}

    prev_recs = json.loads(PREV_SUMMARY.read_text())
    prev_by_key = {(r["instance"], r["solver"]): r for r in prev_recs}

    lines: list[str] = []
    lines.append("# G-set 7問ベンチマーク：前後スカート付き実音・仮想音響 × 古典パラメトロン・Duffing位相変調")
    lines.append("")
    lines.append("## 1. 到達カット値 & BKS到達率（%）比較表")
    lines.append("")
    lines.append("| Instance ($N$, 辺数, **BKS**) | **従来実音 OIM** (B=1) | **新方式 実音** (`B=1`) | **新方式 実音** (`B=8`) | **新方式 仮想音響** (`B=1`) | **新方式 仮想音響** (`B=8`) | 従来仮想音響 (B=8) | 古典 Greedy (32試行) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for g in INST_ORDER:
        bks = int(manifest[g]["bks"])
        m = manifest[g]
        r_old_real = prev_by_key[(g, "real_acoustic_1")]
        r_new_real_1 = by_key.get((g, "real_audio", 1))
        r_new_real_8 = by_key.get((g, "real_audio", 8))
        r_new_virt_1 = by_key[(g, "virtual_audio", 1)]
        r_new_virt_8 = by_key[(g, "virtual_audio", 8)]
        r_old_virt_8 = prev_by_key[(g, "virtual_acoustic_b8")]
        r_greedy = prev_by_key[(g, "greedy_b32")]

        s_real_8 = f"**{int(r_new_real_8['best_cut']):,} ({r_new_real_8['bks_percent']:.2f}%)**" if r_new_real_8 else "-"

        row = [
            f"**`{g}`** ({m['n']:,}, {m['m']:,}, **{bks:,}**)",
            f"{int(r_old_real['best_cut']):,} ({r_old_real['bks_percent']:.2f}%)",
            f"**{int(r_new_real_1['best_cut']):,} ({r_new_real_1['bks_percent']:.2f}%)**",
            s_real_8,
            f"{int(r_new_virt_1['best_cut']):,} ({r_new_virt_1['bks_percent']:.2f}%)",
            f"{int(r_new_virt_8['best_cut']):,} ({r_new_virt_8['bks_percent']:.2f}%)",
            f"{int(r_old_virt_8['best_cut']):,} ({r_old_virt_8['bks_percent']:.2f}%)",
            f"{int(r_greedy['best_cut']):,} ({r_greedy['bks_percent']:.2f}%)",
        ]
        lines.append("| " + " | ".join(row) + " |")

    avg_old_real = statistics.mean(prev_by_key[(g, "real_acoustic_1")]["bks_percent"] for g in INST_ORDER)
    avg_new_real_1 = statistics.mean(by_key[(g, "real_audio", 1)]["bks_percent"] for g in INST_ORDER)
    avg_new_virt_1 = statistics.mean(by_key[(g, "virtual_audio", 1)]["bks_percent"] for g in INST_ORDER)
    avg_new_virt_8 = statistics.mean(by_key[(g, "virtual_audio", 8)]["bks_percent"] for g in INST_ORDER)
    avg_old_virt_8 = statistics.mean(prev_by_key[(g, "virtual_acoustic_b8")]["bks_percent"] for g in INST_ORDER)
    avg_greedy = statistics.mean(prev_by_key[(g, "greedy_b32")]["bks_percent"] for g in INST_ORDER)

    lines.append(
        f"| **7問平均 BKS比** | `{avg_old_real:.2f}%` | **`{avg_new_real_1:.2f}%`** | - | **`{avg_new_virt_1:.2f}%`** | **`{avg_new_virt_8:.2f}%`** | `{avg_old_virt_8:.2f}%` | `{avg_greedy:.2f}%` |"
    )

    lines.append("")
    lines.append("## 2. 実行時間（秒） & Peak RSS（MiB）比較")
    lines.append("")
    lines.append("| Instance ($N$) | **新方式 実音 B=1** 秒 / RSS | **新方式 仮想音響 B=1** 秒 / RSS | **新方式 仮想音響 B=8** 秒 / RSS | 従来仮想音響 B=8 秒 / RSS |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for g in INST_ORDER:
        r_r1 = by_key[(g, "real_audio", 1)]
        r_v1 = by_key[(g, "virtual_audio", 1)]
        r_v8 = by_key[(g, "virtual_audio", 8)]
        r_ov8 = prev_by_key[(g, "virtual_acoustic_b8")]
        lines.append(
            f"| **`{g}`** ({manifest[g]['n']:,}) | {r_r1['seconds']:.1f}s / {r_r1['peak_rss_mib']:.1f} MiB | {r_v1['seconds']:.3f}s / {r_v1['peak_rss_mib']:.1f} MiB | {r_v8['seconds']:.3f}s / {r_v8['peak_rss_mib']:.1f} MiB | {r_ov8['seconds']:.3f}s / {r_ov8['peak_rss_mib']:.1f} MiB |"
        )

    out_file = THIS_DIR / "DUFFING_GSET_REPORT.md"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
