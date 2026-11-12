# The distillation pipeline

How a large teacher voice becomes a small real-time student, and which command does each step. All commands run from the repository root; outputs go to `outputs/` and `data/` (both ignored by git).

```
sentences ──► teacher reads them ──► checks + retries ──► training manifest ──► train student
 texts/        VoxCPM2, one voice      voice / words / pace    English in 2 scripts    VITS, ~36M
                                                                                          │
           listening tests ◄── guardrails + error rates on the benchmark ◄── checkpoints ─┘
```

## 1. Sentences

Write or collect sentences into `texts/`, register the files under a version in `scripts/distill_text.py`, then:

```bash
uv run python scripts/distill_text.py --version v5      # → outputs/distill/v5/tts_in.jsonl
```

Numbers and dates are spelled out here (`samleng.text.verbalize`). The script stops on any sentence that overlaps the benchmark or contains a held-out English word. Have a native speaker spot-check a sample first (`scripts/review_bench.py`).

## 2. Teacher audio

```bash
cd runners/tts_voxcpm
uv run python synthesize.py ../../outputs/distill/v5/tts_in.jsonl ../../outputs/distill/v5/voxcpm2 \
    --reference-wav ../../outputs/voice/teacher_clear.wav --hints ../../texts/teacher_hints.json --resume
```

The teacher clones one reference voice for every sentence, seeded per sentence, and `--resume` skips finished clips. A batch can name its own target voice in `outputs/distill/<batch>/voice.json`.

## 3. Checks and retries

```bash
uv run python scripts/distill_prepare.py prepare --version v5   # 22.05 kHz, trimmed, leveled, voice similarity
uv run python scripts/distill_prepare.py asr --version v5       # ASR round trip, then keep/reject + stats.md
uv run python scripts/distill_prepare.py retry --version v5 --seed 1   # re-record rejects with another seed
```

A clip is rejected when its voice drifts from the reference (`voice`), the recognizer hears different words (`misread`), its speaking rate is an outlier (`speed`), or a native listener flagged it (`ear`, from `texts/teacher_ear_rejects.txt`). Rejected clips are re-recorded rather than dropped, because hard sentences fail most often. Listen to a sample of every batch: the recognizer cannot hear every pronunciation error.

## 4. Training manifest

```bash
uv run python scripts/train_manifest.py v1c v2c v3c v4c v5c --out outputs/train/my_run/manifest.jsonl
```

Joins the kept clips of several batches. A clip with English words gets a second row with the same audio and the words in Khmer script, so the student learns both the dictionary path and raw Latin letters.

## 5. Train

```bash
cd runners/train_vits
uv run python train.py ../../outputs/train/my_run/manifest.jsonl ../../outputs/train/my_run \
    --device mps --batch-size 16 --epochs 30 --eval-clips 100
# resume after a stop
uv run python train.py <manifest> <out> --continue-run <out>/<run folder> --device mps --epochs 30
```

Fine-tunes the base Coqui VITS voice (credited in the README). Held-out clips never share audio with training rows.

## 6. Score checkpoints

```bash
uv run python scripts/eval_checkpoint.py outputs/train/my_run/<run folder> --step 2000
uv run python scripts/eval_checkpoint.py <run folder> --step 2000 --translit          # English fed in Khmer script
uv run python scripts/eval_checkpoint.py <run folder> --step 2000 --tag seed1 "--synth=--noise-scale-dp 0 --seed 1"
uv run python scripts/eval_checkpoint.py <run folder> --voice teacher_clear/0 --pace 0.867   # other target voice and pace
```

Each call voices the 300 Khmer benchmark sentences, transcribes them, and writes `outputs/train/eval/<run>/summary.md`: paired error rates against the base model per category (Wilcoxon), then guardrails (voice similarity, speech-only duration, silence, clipping). Results are cached per step; a folder remembers its synthesis settings and refuses to be reused with others. `ops/watch_run.sh` does this automatically while training.

When a training manifest sits next to the run folder, the summary also has a `mixed_unseen_strict` row (never-seen English words that are also absent in Khmer script from the training text) and ends with the `--length-scale` that would hit the target pace.

To choose between candidate checkpoints, compare them over several seeds:

```bash
uv run python scripts/pick_checkpoint.py <run folder> --steps 18000 20000 --synth="--noise-scale-dp 0 --length-scale 1.3"
```

Rules of thumb from our runs: one seed moves the all-Khmer median by about a point, so use three seeds before comparing; never stop on a single checkpoint; set speaking pace at playback (`--length-scale`), not by re-recording.

## 7. Listen and rate

```bash
uv run python scripts/listen_page.py sentences.jsonl page.html "Base=dir1" "Student=dir2"   # side-by-side page
uv run python scripts/mos.py serve --study my_study --ids ids.txt --system base=... --system student=...
uv run python scripts/mos.py report --study my_study                                         # MOS, CI, Wilcoxon
uv run python scripts/ab.py serve --study my_ab --a student=... --b gemini=... --ids ids.txt  # paired A/B
```

Ratings are blind: clips get opaque ids and a per-rater order. Scores are averaged per sentence first, since ratings of one sentence are not independent, and MOS values are only compared within one study.

## Scripts

| Script | Purpose |
|---|---|
| `baseline.py` | Run TTS systems on a benchmark file and score them (leaderboard) |
| `eval_checkpoint.py` | Score a fine-tuning checkpoint against the base model, with guardrails |
| `pick_checkpoint.py` | Compare candidate checkpoints over several seeds with paired tests |
| `distill_text.py` | Build the teacher's input from `texts/` with leak checks |
| `wiki_text.py`, `tutor_text.py` | Collect Khmer Wikipedia sentences; generate maths and word-drill sentences |
| `distill_prepare.py` | Prepare, check and retry teacher clips |
| `train_manifest.py` | Join checked batches into one training manifest |
| `prepare_data.py` | Clean a real-speech corpus (OpenSLR 42) with an ASR round trip |
| `mos.py`, `ab.py` | Serve blind rating pages and report statistics |
| `validate_metrics.py` | Correlate automatic metrics with human ratings |
| `spk.py` | Voice consistency of each system |
| `listen_page.py`, `review_bench.py`, `review_spellings.py` | Browser pages for listening and for reviewing Khmer text and spellings |
| `phrase_breaks.py` | Train and test the optional phrase-space model |
