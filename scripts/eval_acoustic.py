import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from warnings import filterwarnings

try:
    from pypesq import pesq
except ImportError:
    print("PESQ is not available. Please install pypesq to use PESQ evaluation.")
    print('\tpip install git+https://github.com/vBaiCai/python-pesq.git')
    pesq = None  # PESQ is not available, handle gracefully
from collections import defaultdict

import torch
import torchaudio
from audiocraft.losses.sisnr import SISNR
from past.models.past_model import PastModel

from utils import set_logger, get_device


filterwarnings("ignore")

logger = logging.getLogger(__name__)


def read_one_wav(path: str, target_sr) -> pd.DataFrame:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.shape[0] == 2:
        wav = wav[:1]
    wav = wav.unsqueeze(0)
    return wav


def eval_acostic_metrics(model_cp_path, manifest_path, output_path):
    if not isinstance(manifest_path, Path):
        manifest_path = Path(manifest_path)
    paths = [json.loads(j)['path'] for j in manifest_path.read_text().splitlines()]
    results = eval_acostic_metrics_batch(model_cp_path, paths, 0)
    save_results(dict(results), output_path)


def save_results(results, output_path):
    if output_path.exists():
        new_results = results
        results = json.loads(output_path.read_text())
        results.update(new_results)
    output_path.write_text(json.dumps(results, indent=4))
    logger.info(f"Results saved to {output_path}")


def eval_acostic_metrics_batch(model_cp_path, paths, i):
    device = get_device(i)

    model = PastModel.from_pretrained(model_cp_path).to(device)

    sisnr = SISNR(sample_rate=model.sample_rate, segment=5, overlap=0.5, epsilon=1e-8)
    result_dict = defaultdict(list)
    tq = tqdm(paths)
    for wav_i, path in enumerate(tq):
        wav = read_one_wav(path, model.sample_rate).to(device)

        with torch.no_grad():
            codes, scale = model.encode(wav)
            reconstracted = model.decode(codes, scale)

        reconstracted = reconstracted[:, :, : wav.shape[2]]
        wav = wav[:, :, : reconstracted.shape[2]]

        sisnr_val = sisnr(reconstracted, wav)
        result_dict['sisnr_res'].append(sisnr_val.item())
        if pesq is not None:
            pesq_val = pesq(wav.squeeze().cpu().numpy(), reconstracted.squeeze().cpu().numpy(), model.sample_rate)
            result_dict['pesq_res'].append(pesq_val)
        if wav_i % 50 == 0:
            avg_sisnr = np.mean(result_dict['sisnr_res'])
            msg = f"Processed {wav_i + 1}/{len(paths)} files. Current SISNR: {avg_sisnr:.2f}"
            if pesq is not None:
                avg_pesq = np.mean(result_dict['pesq_res'])
                msg += f", PESQ: {avg_pesq:.2f}"
            logger.info(msg)
    return result_dict


def main(model_cp, output_path, acoustic_menifest):
    set_logger(logger)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_cp).stem if model_cp.endswith('.th') else model_cp
    json_output_path = output_path / f"{model_name}_acoustic_eval.json"
    eval_acostic_metrics(model_cp, acoustic_menifest, json_output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cp", type=str, help="Path or name of the model checkpoint")
    parser.add_argument("--output-path", type=Path, help="Path to the output dir")
    parser.add_argument("--acoustic-menifest", type=Path, help="Path to the acoustic manifest file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(**parse_args().__dict__)
