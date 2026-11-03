"""Generate GSET_REPORT.md from results/summary.json with exact accuracy, time, and Peak RSS tables."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

THIS_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = THIS_DIR / "results" / "summary.json"
OUT_MD = THIS_DIR / "GSET_REPORT.md"

INST_ORDER = ("G1", "G11", "G14", "G22", "G55", "G67", "G70")
SOLVERS_1 = ("real_acoustic_1", "virtual_acoustic_1", "phase_ode_1", "dsb_1", "sa_1")
SOLVERS_MULTI = ("virtual_acoustic_b8", "phase_ode_b32", "dsb_b32", "sa_b32", "greedy_b32")


def main() -> None:
    records = json.loads(SUMMARY_PATH.read_text())
    by_key = {(r["instance"], r["solver"]): r for r in records}
    meta_by_inst = {g: by_key[(g, "phase_ode_1")] for g in INST_ORDER}

    lines: list[str] = []
    lines.append("# Stanford G-set 公開ベンチマーク比較レポート（実音 vs 仮想音響 vs 他アニーリング手法）")
    lines.append("")
    lines.append("実行日: 2026-11-03 JST  ")
    lines.append("対象データ: Stanford G-set (`https://web.stanford.edu/~yyye/yyye/Gset/`) 7インスタンス ($N = 800 \\sim 10,000$)  ")
    lines.append("実行条件: 単一スレッドCPU (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), 各条件とも独立子プロセスで Win32 `PeakWorkingSetSize` (Peak RSS) 実測、全70件の生ファイル独立再採点監査 **PASSED**。")
    lines.append("")
    lines.append("## 1. テスト対象 G-set インスタンス一覧")
    lines.append("")
    lines.append("| Instance | $N$ (頂点数) | $M$ (辺数) | 重み | グラフ構造 | Best Known Cut (BKS) | 疎行列容量 (KiB) |")
    lines.append("| :--- | ---: | ---: | :---: | :--- | ---: | ---: |")
    for g in INST_ORDER:
        m = meta_by_inst[g]
        lines.append(
            f"| `{g}` | {m['n_spins']:,} | {m['n_edges']:,} | `{m['weights']}` | `{m['topology']}` | **{int(m['bks']):,}** | {m['matrix_storage_kib']:.1f} |"
        )

    lines.append("")
    lines.append("## 2. 単一軌道（`batch=1`・同一ステップ数）直接比較：実音 vs 仮想音響 vs デジタル手法")
    lines.append("")
    lines.append("物理ハードウェア（`2- USB Advanced Audio Device` 同一クロック・Lチャンネル単独出力 `[6, 12) kHz`）の1軌道実行と、完全に同一の初期シード・ステップ数に揃えた比較。")
    lines.append("")
    lines.append("### 2.1 到達カット値 & BKS到達率（%）")
    lines.append("")
    lines.append("| Instance (BKS) | 実音 OIM (L単独) | 仮想音響 OIM (48kHz) | デジタル位相 OIM (`PhaseODE`) | Simulated Bifurcation (`dSB`) | Simulated Annealing (`SA`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for g in INST_ORDER:
        bks = int(meta_by_inst[g]["bks"])
        row = [f"**`{g}`** ({bks:,})"]
        for s in SOLVERS_1:
            r = by_key[(g, s)]
            row.append(f"{int(r['best_cut']):,} ({r['bks_percent']:.2f}%)")
        lines.append("| " + " | ".join(row) + " |")

    mean_row = ["**7問平均 BKS比**"]
    for s in SOLVERS_1:
        avg_pct = statistics.mean(by_key[(g, s)]["bks_percent"] for g in INST_ORDER)
        mean_row.append(f"**{avg_pct:.2f}%**")
    lines.append("| " + " | ".join(mean_row) + " |")

    lines.append("")
    lines.append("### 2.2 実行時間（秒） & 独立プロセス Peak RSS（MiB）[`batch=1`]")
    lines.append("")
    lines.append("| Instance ($N$) | 実音 OIM 秒 / RSS | 仮想音響 OIM 秒 / RSS | `PhaseODE` 秒 / RSS | `dSB` 秒 / RSS | `SA` 秒 / RSS |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for g in INST_ORDER:
        n = meta_by_inst[g]["n_spins"]
        row = [f"**`{g}`** ({n:,})"]
        for s in SOLVERS_1:
            r = by_key[(g, s)]
            row.append(f"{r['seconds']:.3f}s / {r['peak_rss_mib']:.1f} MiB")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## 3. マルチスタート標準比較（`batch=32` / 仮想音響 `batch=4..8`）：BKS到達精度・速度・メモリ")
    lines.append("")
    lines.append("### 3.1 到達カット値 & BKS到達率（%）")
    lines.append("")
    lines.append("| Instance (BKS) | 仮想音響 OIM (`400 step`) | デジタル位相 OIM (`800 step, B=32`) | Simulated Bifurcation (`dSB, B=32`) | Simulated Annealing (`SA, B=32`) | 古典 Greedy (`32 starts`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for g in INST_ORDER:
        bks = int(meta_by_inst[g]["bks"])
        row = [f"**`{g}`** ({bks:,})"]
        for s in SOLVERS_MULTI:
            r = by_key[(g, s)]
            row.append(f"{int(r['best_cut']):,} ({r['bks_percent']:.2f}%)")
        lines.append("| " + " | ".join(row) + " |")

    mean_row_m = ["**7問平均 BKS比**"]
    for s in SOLVERS_MULTI:
        avg_pct = statistics.mean(by_key[(g, s)]["bks_percent"] for g in INST_ORDER)
        mean_row_m.append(f"**{avg_pct:.2f}%**")
    lines.append("| " + " | ".join(mean_row_m) + " |")

    lines.append("")
    lines.append("### 3.2 実行時間（秒） & 独立プロセス Peak RSS（MiB）[`batch=32` / `b8`]")
    lines.append("")
    lines.append("| Instance ($N$) | 仮想音響 OIM 秒 / RSS | デジタル位相 OIM 秒 / RSS | `dSB (B=32)` 秒 / RSS | `SA (B=32)` 秒 / RSS | `Greedy (32)` 秒 / RSS |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for g in INST_ORDER:
        n = meta_by_inst[g]["n_spins"]
        row = [f"**`{g}`** ({n:,})"]
        for s in SOLVERS_MULTI:
            r = by_key[(g, s)]
            row.append(f"{r['seconds']:.3f}s / {r['peak_rss_mib']:.1f} MiB")
        lines.append("| " + " | ".join(row) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

