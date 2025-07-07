# Audio Dataset Preparation Guide

This guide walks you through the process of preparing audio datasets (LibriSpeech and TIMIT) for model training, including the generation of aligned transcriptions using a pre-trained Wav2Vec2 model. The output is a structured manifest directory ready for use in downstream training tasks.

---

## 🚀 Overview

The provided script performs the following steps:

1. **Scans** audio files under the provided dataset roots.
2. **Optionally aligns** transcriptions using `facebook/wav2vec2-base-960h`.
3. **Saves** audio metadata and alignment files to a manifest directory.

The output will contain:

* `*.jsonl` files for `train`, `dev`, and `test` splits
* A `transcriptions_alignment/` directory with aligned character timestamps

---

## 📥 Dataset Download

You must manually download and extract the datasets before using the script.

| Dataset     | Link                                                                      |
| ----------- | ------------------------------------------------------------------------- |
| LibriSpeech | [Official Site](https://www.openslr.org/12)                               |
| TIMIT       | [LDC Catalog (LDC93S1)](https://catalog.ldc.upenn.edu/LDC93S1) (licensed) |

---

## 📂 Directory Structure

Expected input structure:

```
<LibriSpeech_root>/train-clean-100/...
<TIMIT_root>/TRAIN/DR1/...
```

Output structure:

```
<output_dir>/
├── LibriSpeech_train.jsonl
├── LibriSpeech_dev.jsonl
├── LibriSpeech_test.jsonl
├── timit_train.jsonl
├── timit_test.jsonl
└── transcriptions_alignment/
    └── [mirrors dataset folder structure]
```

---

## ⚙️ Script Usage

### Basic Command:

```bash
python scripts/audio_dataset_extraction.py \
  --LibriSpeech_root /path/to/LibriSpeech \
  --timit_root /path/to/TIMIT \
  --output_dir /path/to/output_dir
```

### Parameters Explained:

| Argument                          | Required | Description                                                          |
| --------------------------------- | -------- | -------------------------------------------------------------------- |
| `--LibriSpeech_root`              | ✔        | Path to extracted LibriSpeech dataset                                |
| `--timit_root`                    | ✖        | Path to TIMIT dataset (required for phoneme head training) |
| `--output_dir`                    | ✔        | Where to save manifest files and alignments                          |
| `--skip_transcriptions_alignment` | ✖        | Speeds up processing, but disables auxiliary head training           |
| `--debug`                         | ✖        | Limits file count for quick testing                                  |
| `--num_processes`                 | ✖        | Number of parallel processes (default: 8)                            |

---

## 🧠 Notes on Alignment

* Alignment uses HuggingFace's `facebook/wav2vec2-base-960h` to compute character-level timestamps.
* Alignment is **required** for training asr auxiliary head.
* Skipping alignment will produce valid `.jsonl` metadata, but alignment files will be missing.
* TIMIT dataset is used for phoneme classification auxiliary task.

---

## 🖥️ Performance Tips

| Setting             | Recommendation                                       |
| ------------------- | ---------------------------------------------------- |
| Machine Type        | Use a machine with **GPU** (preferably multi-GPU)    |
| Number of Processes | Use a **high number** (e.g., 8-32) for speed         |
| Alignment Skipped   | Much faster, but no support forasr auxiliary head    |

---

## ✅ Example Command With Alignment

```bash
python scripts/audio_dataset_extraction.py \
  --LibriSpeech_root /data/LibriSpeech \
  --timit_root /data/TIMIT \
  --output_dir /data/manifests \
  --num_processes 16
```

## ⚡ Example Command Without Alignment

```bash
python scripts/audio_dataset_extraction.py \
  --LibriSpeech_root /data/LibriSpeech \
  --output_dir /data/manifests \
  --skip_transcriptions_alignment \
  --num_processes 16
```

---

## 📌 FAQ

**Q: Can I run with just LibriSpeech?**
A: Yes. But phoneme alignment head training will not be possible.

**Q: What if I skip `--skip_transcriptions_alignment`?**
A: The script will generate transcriptions using Wav2Vec2. This is slower but enables alignment-based training.

**Q: What model is used for alignment?**
A: `facebook/wav2vec2-base-960h` from HuggingFace Transformers.
