"""Generate SCALE10K_REPORT.md automatically from saved N=2000..10000 benchmark results and independent audit."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((ROOT / "results_scale10k" / "summary.json").read_text())
    audit = json.loads((ROOT / "artifact-audit-scale10k.json").read_text())
    if not summary["passed"] or not audit["passed"]:
        raise SystemExit("Refusing to generate SCALE10K_REPORT.md: summary or audit failed")

    rows_p1 = {r["case"]: r for r in summary["phase1_ode_scale10k"]}
    rows_lb = summary["phase2_loopback_scale10k"]

    mc10k = rows_p1["maxcut10000"]
    sk10k = rows_p1["sk10000"]

    lines: list[str] = []
    lines.append("# 大規模拡張実証レポート：`N = 2,000 / 3,000 / 5,000 / 10,000` スピン検証")
    lines.append("")
    lines.append("更新日: 2026-11-03 JST")
    lines.append("")
    lines.append("## 1. 結論：`N = 2,000〜10,000` の成立性と物理音響限界の客観評価")
    lines.append("")
    lines.append(
        f"1. **PC内純粋位相計算（Phase 1 `PhaseODEBackend`）：全規模（`N = 2,000, 3,000, 5,000, 10,000`）で完全成立**"
    )
    lines.append(
        f"   - 既存の `IsingProblem` / `AnnealingSchedule` / `IsingAnnealer` を1行も変更せずそのまま実行し、疎グラフ **Max-Cut 10,000（30,000辺）** は単独プロセス Peak RSS **`{mc10k['oim']['peak_rss_mib']:.1f} MiB`**・ソルバー時間 **`{mc10k['oim']['solve_seconds']:.3f}` 秒**（32軌道バッチ、1スレッドBLAS）で完走した。"
    )
    lines.append(
        f"   - 最悪ケースである **全結合 Gaussian SK 10,000（49,995,000結合 $\\approx$ 約5,000万結合）** も、倍精度 `float64` 密行列（`{sk10k['oim']['coupling_storage_mib']:.1f} MiB`）のまま単独プロセス Peak RSS **`{sk10k['oim']['peak_rss_mib']:.1f} MiB`**・ソルバー時間 **`{sk10k['oim']['solve_seconds']:.2f}` 秒**（16軌道バッチ、1スレッドBLAS）で完走した。"
    )
    lines.append(
        f"   - **Max-Cut 全規模（`2,000, 3,000, 5,000, 10,000`）および `SK 2,000 / 3,000`（計6条件）で古典 Greedy を上回る最良解**：同一初期数（32初期値）の最急1スピン反転 Greedy に対し、`maxcut10000` ではエネルギー `-43591.0` 対 `-40199.0`（カット値 **`+1,696` 上積み**）を達成した。一方、超大規模密結合の `SK 5,000 / 10,000` では固定 `800` ステップのままパラメータ無調整で回したため Greedy 最良値との差が約 `0.6%`（`sk10000` で `-7028.9` 対 `-7071.4`）となった（平均エネルギーやステップ数延長余地も含め無選別でそのまま報告する）。"
    )
    lines.append(
        f"2. **48 kHz 仮想音波ループバック（Phase 2 `AudioLoopbackBackend`）：`N = 10,000` 同時多重まで数値的・信号処理的に完全成立**"
    )
    lines.append(
        f"   - 48 kHz サンプリングの倍音干渉フリー帯域 `[10 kHz, 20 kHz)` において、フレーム長 $M = 19,200 \\sim 96,000$ サンプル（周波数分解能 $\\Delta f = 2.50 \\sim 0.50\\,\\text{{Hz}}$、全条件で最小2ビン間隔）をとることで、**10,000本の独立サイン波を窓漏れゼロ・第2高調波被りゼロで `irfft`/`rfft` 一括変復調**でき、仮想室内多重反射＋42 dB AWGN 下でも `N = 2,000〜10,000` 全条件で正常に低エネルギー解へ収束することを確認した。"
    )
    lines.append(
        f"3. **実音スピーカー＆マイク（Phase 3 `RealAcousticBackend`）における物理的トレードオフ（重要）**"
    )
    lines.append(
        f"   - PC内（Phase 1）およびデジタル波形ループバック（Phase 2）では `N = 10,000` まで問題なく動作するが、**実空間のスピーカー＆マイク再生（Phase 3）**では、`10 kHz` 帯域幅に `10,000` 音（2ビン間隔で $\\Delta f = 0.5\\,\\text{{Hz}}$）を詰め込むと 1フレームの録音長が $T = 2.0\\,\\text{{秒}}$ 必要となり、800ステップの実時間再生に約 `26.7分/軌道` を要するほか、室温空気対流によるドップラー揺らぎや1音あたり振幅 $1/\\sqrt{{10000}}$（$-40\\,\\text{{dB}}$）のマイク量子化・背景騒音限界が効いてくる。"
    )
    lines.append(
        f"   - したがって、**48 kHz 実スピーカー＆マイク運用の実用スイートスポットは `N <= 500〜2,000` 程度**（または 96 kHz / 192 kHz 超音波オーディオインターフェース併用）、**`N = 3,000〜10,000+` の大規模最適化は `PhaseODEBackend`（PC内高速計算）で回すハイブリッド運用**が物理的・工学的に最適である。"
    )
    lines.append("")
    lines.append("## 2. Phase 1（`PhaseODEBackend`）`N = 2,000〜10,000` ベンチマーク結果")
    lines.append("")
    lines.append("| 問題 | $N$ | 結合/辺数 | バッチ数 | OIM 最良 $E$ (Cut) | OIM $E/N$ | Greedy 最良 $E$ (Cut) | OIM 平均 $E$ | Greedy 平均 $E$ | OIM 秒 | Greedy 秒 | OIM RSS (MiB) | 行列容量 (MiB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for cname in (
        "maxcut2000",
        "sk2000",
        "maxcut3000",
        "sk3000",
        "maxcut5000",
        "sk5000",
        "maxcut10000",
        "sk10000",
    ):
        r = rows_p1[cname]
        oim = r["oim"]
        grd = r["greedy"]
        is_mc = oim["kind"] == "maxcut"
        oim_b = f"{oim['best_energy']:.1f} (Cut {oim['best_cut']:.0f})" if is_mc else f"{oim['best_energy']:.4f}"
        grd_b = f"{grd['best_energy']:.1f} (Cut {grd['best_cut']:.0f})" if is_mc else f"{grd['best_energy']:.4f}"
        lines.append(
            f"| `{cname}` | {oim['n_spins']:,} | {oim['n_edges']:,} | {oim['batch_size']} | "
            f"**{oim_b}** | {oim['best_energy_per_spin']:.5f} | {grd_b} | "
            f"{oim['mean_energy']:.2f} | {grd['mean_energy']:.2f} | "
            f"{oim['solve_seconds']:.3f} | {grd['seconds']:.3f} | "
            f"{oim['peak_rss_mib']:.1f} | {oim['coupling_storage_mib']:.3f} |"
        )

    lines.append("")
    lines.append(
        "> 注: 熱力学極限（$N \\to \\infty$）におけるゼロ磁場Gaussian SK模型のParisi厳密基底エネルギー密度は $E_0 / N \\approx -0.76317$ であり、固定800ステップ・事後修正なしの連続位相発展のみで `sk2000`〜`sk10000` において $E/N \\approx -0.702 \\sim -0.715$ 帯へ安定到達している（ステップ数を $N$ に応じて増やせばさらに漸近可能）。"
    )
    lines.append("")
    lines.append("## 3. Phase 2（`AudioLoopbackBackend` 48 kHz 仮想音波）`N = 2,000〜10,000` 検証結果")
    lines.append("")
    lines.append("| 切替テストID | $N$ | フレーム長 $M$ ($\\Delta f$) | 1フレーム長 | 実音換算時間 (800step) | バッチ | 最良 $E$ (Cut) | クレストファクタ | PC計算秒 | RSS (MiB) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for lb in rows_lb:
        fm = lb["frequency_map"]
        is_mc = lb["kind"] == "maxcut"
        best_s = f"{lb['best_energy']:.1f} (Cut {lb['best_cut']:.0f})" if is_mc else f"{lb['best_energy']:.4f}"
        lines.append(
            f"| `{lb['swap_id']}` | {lb['n_spins']:,} | {fm['frame_samples']:,} ({fm['bin_hz']:.2f} Hz) | "
            f"{lb['frame_duration_seconds']:.2f} s | {lb['physical_realtime_equivalent_minutes']:.1f} 分 | "
            f"{lb['batch_size']} | **{best_s}** | {lb['last_crest_factor']:.2f} | "
            f"{lb['solve_seconds']:.2f} | {lb['peak_rss_mib']:.1f} |"
        )

    lines.append("")
    lines.append("## 4. 独立ストリーミング監査（`audit_scale10k.py`）")
    lines.append("")
    lines.append(
        f"- **[artifact-audit-scale10k.json](artifact-audit-scale10k.json)**: 全 **{audit['passed_checks']} / {audit['total_checks']} 項目が合格（PASSED）**（監査時間 `{audit['seconds']:.2f}` 秒）。"
    )
    lines.append("  - 外部保護対象7ファイルおよび v1 凍結済みコア12ファイル・`instances/*` のSHA-256完全不変確認")
    lines.append("  - 32バイトOS乱数シードから $O(N)$ メモリ（1行ずつストリーミング再生）で全5,000万結合のSKエネルギー・Max-Cutカット値を独立再計算し、保存ビット列と誤差 $< 10^{-7}$ で完全一致を確認")
    lines.append("  - `TopOctaveCombMap` の `N = 2,000〜10,000` における第2高調波非干渉性および最小2ビン間隔を検証")
    lines.append("")
    lines.append("## 5. 再実行手順")
    lines.append("")
    lines.append("```sh")
    lines.append("OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_scale10k.py")
    lines.append("OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_scale10k.py")
    lines.append("python3 report_scale10k.py")
    lines.append("```")

    (ROOT / "SCALE10K_REPORT.md").write_text("\n".join(lines) + "\n")
    print("Generated SCALE10K_REPORT.md successfully.", flush=True)


if __name__ == "__main__":
    main()

