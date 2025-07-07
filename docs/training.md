# Training the PAST Model

This guide will walk you through the process of training the `PAST` model from scratch.

---

## 📥 1. Clone the Repository

```bash
git clone https://github.com/slp-rl/PAST.git
cd PAST
```

---

## 🛠️ 2. Set Up the Environment

Follow the instructions in the [main README](../README.md) to set up the Python environment and install dependencies.

---

## 📁 3. Prepare the Dataset

You must preprocess and align your dataset before training. Follow the detailed steps in our [Data Preparation Guide](./data_preparation.md) to generate the necessary manifests.

---

## 🎛️ 4. Configuration and Framework

Training is based on the [Audiocraft](https://audiocraft.metademolab.com) framework. You can refer to their [training documentation](https://github.com/facebookresearch/audiocraft/blob/main/docs/TRAINING.md) for additional guidance on architecture and workflow.

All configuration values (with defaults) can be found in our example config file:
[Example Configuration](../assets/past_config.yaml).

You may override any of these parameters directly from the command line.

---

## 🚀 5. Basic Training Command

Run the following command to start training:

```bash
dora run solver=compression/rvq_transformer_tasks.yaml datasource.manifests_dir=<YOUR_MANIFESTS_DIR>
```

change `dora run` to `dora run -d` if you want to run it in multi-device mode.

---

## 🧠 6. Enabling Auxiliary Heads

To enable ASR and phoneme quantization auxiliary heads, add the following flags:

```bash
auxiliary_tasks.asr_quant.apply=true auxiliary_tasks.phone_quant.apply=true
```
> ℹ️ **Note:** In default configuration, auxiliary heads are disabled.

---


## 🔄 7. Resuming from a Checkpoint or Pretrained Model

You can resume training from a previous checkpoint or a model hosted on HuggingFace:

- From local checkpoint:

```bash
continue_from=<YOUR_MODEL_CP>
```

- From HuggingFace:

```bash
continue_from=PAST
```

> ℹ️ **Note:** You can fine-tune the model **with or without auxiliary heads**, even when loading a checkpoint that includes them.  
> If you choose not to enable the auxiliary heads during training, they will simply not be loaded, and you'll receive a warning that parts of the model were not restored. This is expected and safe.

---

## 🧪 8. Debug Training Mode (Quick Tests)

To speed up training and debugging, use the following flags:

```bash
optim.epochs=10 \
optim.updates_per_epoch=100 \
dataset.batch_size=16 \
dataset.valid.num_samples=16 \
dataset.num_workers=0 \
generate.every=2 \
evaluate.every=2
```

---

## 🔄 9. Streamable Variant

To train with causal (streamable) configuration:

```bash
encodec.causal=true seanet.lstm_bidirectional=false
```

Use the pretrained streamable model as a starting point:

```bash
continue_from=PAST_streamable
```

---

## 📊 10. Logging with Weights & Biases (W&B)

The model supports logging to [Weights & Biases](https://docs.wandb.ai/quickstart/).

To enable W&B integration, make sure to:
- Log in to W&B via `wandb login`
- Set environment variables (optional) for your project and entity

---

Happy training!