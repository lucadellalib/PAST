# Evaluation Guide for PAST

This guide provides instructions for evaluating the PAST model using the metrics reported in the paper.

---

## ⚙️ Setup

Before running the evaluation:

1. 📥 **Clone the repository**
```bash
git clone https://github.com/slp-rl/PAST.git
cd PAST
```

2. 🛠️ **Set up the environment**  
Follow the instructions in the [main README](../README.md)

3. 📁 **Prepare the dataset**  
See our [Data Preparation Guide](./data_preparation.md)  
> ✅ Note: For evaluation only, you **do not need** transcription alignments — this makes the process much faster!

---

## 📐 Evaluation Metrics (Brief Overview)

The PAST model is evaluated on both acoustic and phonetic criteria:

- **SISNR** (Scale-Invariant Signal-to-Noise Ratio): Measures fidelity of waveform reconstruction.
- **PESQ** (Perceptual Evaluation of Speech Quality): Quantifies perceptual speech quality.
- **ViSQOL**: Approximates human MOS scores using perceptual similarity.
- **PNMI** (Phone-Normalized Mutual Information): Measures how informative the token sequence is about the phonemes.
- **ABX**: Evaluates whether phonetic distinctions are preserved after tokenization.
- **WER** (Word Error Rate): Evaluates ASR quality based on discrete tokens.
- **sWUGGY**: Evaluates speech-language model performance.

---

## 🎧 Acoustic Metrics: SISNR & PESQ

Use this script to compute reconstruction metrics:

```bash
python scripts/eval_acoustic.py \
  --model-cp PAST \
  --output-path <PATH_TO_OUTPUT_DIR> \
  --acoustic-menifest <YOUR_MANIFESTS_DIR>/LibriSpeech_test.jsonl
```

You can replace `PAST` with:
- A different HuggingFace model (e.g. `PAST_streamable`)
- A path to a local checkpoint

You can also use any manifest you wish, as long as it contains audio files, and in Audiocraft format.

> ⚠️ **Note:** To compute the **PESQ** metric, you must install the `pypesq` dependency manually, as it's not included in the project's environment by default.  
> Run the following command inside your conda environment:
> ```bash
> pip install git+https://github.com/vBaiCai/python-pesq.git
> ```
> If you skip this step, the evaluation will still run, but **PESQ will not be computed**.
---

## 🔤 Phonetic Metric: PNMI

Use this script to compute PNMI using phoneme labels:

```bash
python scripts/eval_phonme.py \
  --model-cp PAST \
  --output-path <PATH_TO_OUTPUT_DIR> \
  --timit-manifest <YOUR_MANIFESTS_DIR>/timit_test.jsonl
```

As before, you can use any manifest and model you wish.

---

## 📝 Word Error Rate (WER)

WER is computed using the [DASB Benchmark](https://github.com/speechbrain/benchmarks/tree/DASB/benchmarks/DASB)

🔗 **Guide for tokenizer integration**:  
[→ Incorporating Your Audio Tokenizer](https://github.com/speechbrain/benchmarks/tree/DASB/benchmarks/DASB#-incorporating-your-audio-tokenizer)

---

## 🧪 ABX Metric

ABX tests how well the tokenizer preserves phoneme distinctions using continuous embeddings.  
We use the official tools from the [Libri-light repo](https://github.com/facebookresearch/libri-light/blob/main/eval/README.md).

In our evaluation, we extracted **reconstructed embeddings** using:

```python
model.decode(codes, scale, return_latent=True)
```

This allows evaluation directly on the latent representations after RVQ.

---

## 🎚️ ViSQOL

ViSQOL is computed using Audiocraft’s wrapper around Google’s implementation.

📘 Documentation:  
[→ Audiocraft Metrics](https://github.com/facebookresearch/audiocraft/blob/main/docs/METRICS.md)

---

## 🧠 sWUGGY Metric

To evaluate spoken language modeling performance, we trained language models over different tokenizers’ outputs.  
We then used the [salmon](https://pages.cs.huji.ac.il/adiyoss-lab/salmon/) library to compute the **sWUGGY** metric.

This benchmark tests the model’s ability to assign higher likelihoods to real words over pseudo-words.

---

Happy evaluating! 🎧📈