# Acoustic Oscillator Ising Machine (`acoustic-ising-v1`)

組合せ最適化問題（重み付きMax-Cut、全結合Gaussian SK模型、一般イジング問題）を $O(N)$ 個の古典連続位相スピン（Phase Spin）の時間発展で解く発振器イジングマシン（OIM）実装。上位のアニーリング制御（`IsingAnnealer`）を1行も書き換えずに、以下の3バックエンドをシームレスに切り替えられる。

1. **`PhaseODEBackend`（Phase 1：PC内純粋位相方程式）**：搬送波合成を行わず、バッチ並列BLAS積のみで高速・省メモリに位相発展を計算（スピーカー無音）。
2. **`AudioLoopbackBackend`（Phase 2：48 kHz 仮想音波ループバック）**：`Musical12ToneMap`（C5〜B5 可聴12音）または `TopOctaveCombMap`（10–20 kHz 倍音干渉フリー高密度コム）により実48 kHzマルチトーン波形を `irfft` 合成し、Schroeder位相オフセット・サイクリックプレフィックス・仮想室内多重反射＋AWGN・`rfft` 復調・$H(f_i)$ キャリブレーションを通して位相を観測（スピーカー無音・WAV書き出し対応）。
3. **`RealAcousticBackend`（Phase 3：実スピーカー＆マイク入出力）**：PortAudio（`sounddevice`）経由でパイロットチャープ相互相関同期＋室内伝達関数 $H(f_i)$ 校正を行う実音バックエンド（誤再生防止のため `allow_speaker_output=True` 指定時のみ有効化）。

## ドキュメント・成果物

- 事前固定プロトコル: [PROTOCOL.md](PROTOCOL.md)
- 実験・ベンチマーク結果レポート: [REPORT.md](REPORT.md)
- 独立監査ログ: [artifact-audit.json](artifact-audit.json)
- 集計サマリ: [results/summary.json](results/summary.json)
- 12音和音ロック過程の試聴用WAV（自動再生なし）: `results/maxcut12-chord-evolution.wav`

## 実行手順（1スレッドBLAS固定）

```sh
# 1. 凍結済みベンチマーク実行（Phase 1 全10条件 + Phase 2 仮想音波切替4条件）
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 run.py

# 2. 保存成果物・生乱数リプレイ・独立スピン再採点・外部ファイル不変性の独立監査
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit.py

# 3. REPORT.md 再生成
python3 report.py
```

