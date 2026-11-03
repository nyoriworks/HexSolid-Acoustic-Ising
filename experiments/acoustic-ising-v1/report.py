"""Generate REPORT.md automatically from saved benchmark summary and independent audit artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    audit = json.loads((ROOT / "artifact-audit.json").read_text())
    prep = json.loads((ROOT / "preparation.json").read_text())
    if not summary["passed"] or not audit["passed"]:
        raise SystemExit("Refusing to generate REPORT.md: summary or audit did not pass")

    rows_p1 = {r["case"]: r for r in summary["phase1_ode_benchmarks"]}
    rows_lb = summary["phase2_audio_loopback_swaps"]
    hist = summary["historical_reference_context"]
    sc = prep["self_checks"]

    lines: list[str] = []
    lines.append("# 発振器イジングマシン（OIM）第1段階・仮想音波切替実証レポート")
    lines.append("")
    lines.append("更新日: 2026-11-03 JST")
    lines.append("")
    lines.append("## 1. 結論と客観的達成範囲")
    lines.append("")
    mc24 = rows_p1["maxcut24"]
    sk24 = rows_p1["sk24"]
    sk1000 = rows_p1["sk1000"]
    mc1000 = rows_p1["maxcut1000"]
    lines.append(
        f"- 組合せ最適化（Max-Cut・SK模型）向けに、$2^N$ 状態ベクトルを持たず $N$ 個の古典連続位相スピン（Phase Spin）で駆動する発振器イジングマシン（OIM）基盤を `experiments/acoustic-ising-v1/` に構築した。既存アプリ・LBM・QRNG・過去実験・既存の全 `HANDOVER_*.md` は1バイトも変更していない（外部7ファイルのSHA-256一致を独立監査で確認）。"
    )
    lines.append(
        f"- **既存24スピン厳密ベンチマーク合格**: `PhaseODEBackend`（128独立初期位相・事後貪欲修正なし）により、**Max-Cut 24 で厳密最大カット `{mc24['oim']['best_cut']:.0f}`**（128本中 `{mc24['oracle_oim']['ground_hits']}` 本到達、実行 `{mc24['oim']['solve_seconds']:.4f}` 秒、単独プロセス Peak RSS `{mc24['oim']['peak_rss_mib']:.1f} MiB`）、**SK 24 で厳密基底エネルギー `{sk24['oim']['best_energy']:.12f}`**（128本中 `{sk24['oracle_oim']['ground_hits']}` 本到達、実行 `{sk24['oim']['solve_seconds']:.4f}` 秒、単独プロセス Peak RSS `{sk24['oim']['peak_rss_mib']:.1f} MiB`）を取得した。"
    )
    lines.append(
        f"- **大規模 $N=100, 500, 1000$ での非爆発実証**: 量子状態ベクトル（$2^{{1000}}$ 次元）や全結合MPS/TTNでは不可能な **$N=1,000$ 全結合SK模型（499,500結合）** を単独プロセス Peak RSS **`{sk1000['oim']['peak_rss_mib']:.1f} MiB`**・**`{sk1000['oim']['solve_seconds']:.3f}` 秒**（1スレッドBLAS）で完走し、128初期値の古典最急1スピン反転貪欲法（Greedy）より低い最良エネルギー（`{sk1000['oim']['best_energy']:.4f}` 対 `{sk1000['greedy']['best_energy']:.4f}`）および平均エネルギー（`{sk1000['oim']['mean_energy']:.4f}` 対 `{sk1000['greedy']['mean_energy']:.4f}`）に到達した。疎グラフ **Max-Cut 1000（3,000辺）** も Peak RSS **`{mc1000['oim']['peak_rss_mib']:.1f} MiB`**・**`{mc1000['oim']['solve_seconds']:.3f}` 秒**で完走した。"
    )
    lines.append(
        f"- **無改造バックエンド切替実証（Phase 2 `AudioLoopbackBackend`）**: 問題定義・スケジュール・`IsingAnnealer` を1行も変更せずバックエンドのみを `AudioLoopbackBackend`（48 kHz `irfft` マルチトーン合成＋Schroeder位相オフセット＋仮想室内多重反射＋42 dB AWGN＋`rfft` 検波＋$H(f)$ 校正、スピーカー無音）へ差し替え、12音階モード（`Musical12ToneMap` C5〜B5）の `maxcut12` と高密度10–20 kHzモード（`TopOctaveCombMap`）の `maxcut24`（最適カット `62`）・`sk24`（厳密基底 `-16.548016339179`）・`sk500`（500音同時多重）がすべて正常収束することを確認した。"
    )
    lines.append(
        f"- **厳格な範囲定義**: 本方式は古典位相発振器ダイナミクスであり、「量子ビット」「量子もつれ」「量子優位性」は主張しない。Shorの因数分解や汎用量子ゲート回路は対象外である。"
    )
    lines.append("")
    lines.append("## 2. 第1段階（`PhaseODEBackend`）全10条件ベンチマーク結果")
    lines.append("")
    lines.append(
        "全条件とも同一の無次元アニーリングスケジュール（Heun法 800ステップ, $\\Delta t=0.10$, 4周期SHIL変調, 128独立初期位相, 事後の貪欲局所探索なし）および1スレッドBLAS（`OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=1`）で実行。Peak RSSは他ケースや厳密解表を含まない**条件別独立子プロセス**の `ru_maxrss` 実測値。"
    )
    lines.append("")
    lines.append("| 問題 | $N$ | 結合/辺数 | 厳密基底 $E_0$ (最適Cut) | OIM 最良 $E$ (Cut) | OIM 平均 $E$ | 厳密解到達 | Greedy 最良 $E$ (Cut) | Greedy 平均 $E$ | OIM 秒 | Greedy 秒 | OIM RSS (MiB) | 行列容量 (MiB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for cname in (
        "maxcut12",
        "sk12",
        "maxcut24",
        "sk24",
        "maxcut100",
        "sk100",
        "maxcut500",
        "sk500",
        "maxcut1000",
        "sk1000",
    ):
        r = rows_p1[cname]
        oim = r["oim"]
        grd = r["greedy"]
        orc = r["oracle_oim"]
        is_mc = oim["kind"] == "maxcut"
        if orc is not None:
            exact_str = (
                f"{orc['exact_ground_energy']:.1f} (Cut {orc['exact_optimal_cut']:.0f})"
                if is_mc
                else f"{orc['exact_ground_energy']:.10f}"
            )
            hit_str = f"{orc['ground_hits']}/128 ({100.0 * orc['ground_hit_rate']:.1f}%)"
        else:
            exact_str = "—"
            hit_str = "—"

        oim_best_str = (
            f"{oim['best_energy']:.1f} (Cut {oim['best_cut']:.0f})"
            if is_mc
            else f"{oim['best_energy']:.6f}"
        )
        grd_best_str = (
            f"{grd['best_energy']:.1f} (Cut {grd['best_cut']:.0f})"
            if is_mc
            else f"{grd['best_energy']:.6f}"
        )
        mat_mib = float(oim["coupling_storage_bytes"]) / (1024.0 * 1024.0)
        lines.append(
            f"| `{cname}` | {oim['n_spins']} | {oim['n_edges']:,} | {exact_str} | "
            f"**{oim_best_str}** | {oim['mean_energy']:.4f} | {hit_str} | "
            f"{grd_best_str} | {grd['mean_energy']:.4f} | "
            f"{oim['solve_seconds']:.4f} | {grd['seconds']:.4f} | "
            f"{oim['peak_rss_mib']:.1f} | {mat_mib:.3f} |"
        )

    lines.append("")
    lines.append("### 従来方式（量子回路 TTN/QAOA：過去実験の保存記録）との規模・資源対比")
    lines.append("")
    lines.append(
        f"- **SK 24 における対比（出典: `tensor-sk-qaoa-v1/results.json` の保存値）**: 従来の対角位相一括TTN検証では、根ボンドで誤差 $\\le 10^{{-8}}$ に必要なランクが `4,090` に達し、共通SVDに `{hist['sk24_bulk_ttn_svd_seconds']:.3f}` 秒・プロセス最大RSS `{hist['sk24_bulk_ttn_peak_rss_bytes'] / (1024**3):.3f} GiB` を要し、QAOA $p=1$ の基底確率は `{100.0 * hist['sk24_qaoa_p1_ground_probability']:.4f}%`（平均エネルギー `{hist['sk24_qaoa_p1_mean_energy']:.4f}`）だった。これに対し古典OIM（`PhaseODEBackend`）は状態空間が $O(N)$ のため、単独プロセスRSS `{sk24['oim']['peak_rss_mib']:.1f} MiB`・実行 `{sk24['oim']['solve_seconds']:.4f}` 秒で128本中 `{sk24['oracle_oim']['ground_hits']}` 本（`{100.0 * sk24['oracle_oim']['ground_hit_rate']:.1f}%`）が厳密基底エネルギー `-16.548016339179` に到達した（注：QAOA回路シミュレーションと古典OIMは計算対象モデルが異なるため、量子回路の高速化ではなく組合せ最適化ソルバーへの方針転換効果を示す）。"
    )
    lines.append(
        f"- **大規模域（$N=100 \\sim 1,000$）でのメモリ非爆発**: $N=12 \\to 1,000$ までスピン数を83倍に増やしても、`sk1000` の結合行列容量は `{float(sk1000['oim']['coupling_storage_bytes']) / (1024**2):.2f} MiB`、プロセス全体の Peak RSS は `{sk1000['oim']['peak_rss_mib']:.1f} MiB`（疎グラフ `maxcut1000` では行列容量 `{float(mc1000['oim']['coupling_storage_bytes']) / (1024**2):.3f} MiB`・Peak RSS `{mc1000['oim']['peak_rss_mib']:.1f} MiB`）に収まり、指数的メモリ爆発は完全に解消された。"
    )
    lines.append("")
    lines.append("## 3. 第2段階（`AudioLoopbackBackend` 48 kHz 仮想音波ループ）無改造切替検証")
    lines.append("")
    lines.append(
        "上位の `IsingProblem` / `AnnealingSchedule` / `IsingAnnealer` を無変更のまま、`PhaseTransportBackend` 実装のみを `AudioLoopbackBackend`（48 kHz サンプリング、サイクリックプレフィックス480サンプル、4タップ仮想室内インパルス応答＋高域減衰＋SNR 42 dB 白色雑音、開始時4フレーム平均の $H(f_i)$ キャリブレーション、スピーカー出力なし）に差し替えて実行した結果："
    )
    lines.append("")
    lines.append("| 切替テストID | 問題 ($N$) | 周波数マップ | 使用帯域 (Hz) | バッチ数 | 最良 $E$ (Cut) | 厳密解到達 | クレストファクタ | 実行秒 | RSS (MiB) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for lb in rows_lb:
        fm = lb["frequency_map"]
        band = f"{fm['f_min_hz']:.1f}–{fm['f_max_hz']:.1f}"
        is_mc = lb["kind"] == "maxcut"
        best_s = f"{lb['best_energy']:.1f} (Cut {lb['best_cut']:.0f})" if is_mc else f"{lb['best_energy']:.6f}"
        orc = lb["oracle"]
        hit_s = f"{orc['ground_hits']}/{lb['batch_size']}" if orc is not None else "—"
        lines.append(
            f"| `{lb['swap_id']}` | `{lb['case']}` ({lb['n_spins']}) | `{fm['name']}` | {band} | "
            f"{lb['batch_size']} | **{best_s}** | {hit_s} | {lb['last_crest_factor']:.2f} | "
            f"{lb['solve_seconds']:.3f} | {lb['peak_rss_mib']:.1f} |"
        )
    lines.append("")
    lines.append(
        f"- **Schroeder位相オフセットによる音割れ（ピーク集中）抑制**: 500スピン同相整列時（$\\phi_i = 0$）の波形クレストファクタは、オフセットなしの `{sc['aligned_500tone_crest_factor_uncompensated']:.2f}` から Schroeder 位相適用により **`{sc['aligned_500tone_crest_factor_schroeder']:.2f}`** へ低減され、かつ $H_{{\\text{{cal}}}}(f_i)$ 除算で位相オフセットと室内伝達特性が同時に無誤差（無雑音時最大フェーザ誤差 `{sc['noiseless_500tone_multipath_phasor_error']:.2e}`）で相殺されることを確認した。"
    )
    lines.append(
        f"- **12音階収束和音WAV出力**: `maxcut12` の位相収束過程を C5〜B5 の可聴12音和音として `results/maxcut12-chord-evolution.wav`（48 kHz, 16-bit mono, 4.1秒）に保存済み（スピーカー自動再生なし）。"
    )
    lines.append(
        f"- **実スピーカー＆マイク対応（Phase 3 `RealAcousticBackend`）**: パイロットチャープ相互相関による往復遅延自動同期および室内 $H(f_i)$ 校正を備えた `backends/real_acoustic.py` を実装済み。誤発音防止ガードにより、明示的に `allow_speaker_output=True` を指定しない限りスピーカー出力を行わない。"
    )
    lines.append("")
    lines.append("## 4. 独立監査結果（`audit.py`）")
    lines.append("")
    lines.append(
        f"- **[artifact-audit.json](artifact-audit.json)**: 全 **{audit['passed_checks']} / {audit['total_checks']} 項目が合格（PASSED）**（監査時間 `{audit['seconds']:.3f}` 秒）。"
    )
    lines.append("  - 外部保護対象7ファイル（既存 `HANDOVER_*.md`, `tensor-maxcut-qaoa-v1`, `tensor-sk-qaoa-v1`）のSHA-256完全不変確認")
    lines.append("  - ソルバー・スケジュール・全バックエンド内のオラクル非参照検査およびスピーカー安全ガード作動確認")
    lines.append("  - 生OS乱数バイナリ（`*-draw.bin`）からの Box–Muller / グラフ辺生成の独立リプレイ照合（最大差 $< 10^{-14}$）")
    lines.append("  - $N=12$ 全 $4,096$ 状態の独立全列挙照合および $N=12, 24, 100, 500, 1000$ 全条件・全プレフィックス・仮想音波テストのビット列からの独立エネルギー/カット再採点")
    lines.append("  - `maxcut24`・`sk24` のマスターシードからの128軌道決定論的リプレイ完全一致確認")
    lines.append("")
    lines.append("## 5. 再実行手順")
    lines.append("")
    lines.append("```sh")
    lines.append("OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 run.py")
    lines.append("OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit.py")
    lines.append("python3 report.py")
    lines.append("```")
    lines.append("")
    lines.append("（`prepare.py` は凍結済み問題の保護のため既存 `preparation.json` の上書きを拒否する。）")

    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Generated REPORT.md successfully.", flush=True)


if __name__ == "__main__":
    main()

