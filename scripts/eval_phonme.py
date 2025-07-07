import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from warnings import filterwarnings

import torch
import torchaudio

from past.models.past_model import PastModel
from scripts.utils import set_logger


filterwarnings("ignore")
logger = logging.getLogger(__name__)


def calc_token_purity(joint_dist, marginal_phone_dist, most_like_token_by_phone):
    purity = 0
    for i, phone_row in marginal_phone_dist.iterrows():
        curr_most_like_token = most_like_token_by_phone.loc[phone_row.name]
        cond_prob = joint_dist.loc[phone_row.name, curr_most_like_token["token"]]["p"] / phone_row["p"]
        purity += phone_row["p"] * cond_prob
    logger.info(f'Phone purity: {purity:.3f}')
    return purity


def calc_phone_purity(joint_dist, marginal_token_dist, most_like_phone_by_token):
    purity = 0
    for i, token_row in marginal_token_dist.iterrows():
        curr_most_like_token = most_like_phone_by_token.loc[token_row.name]
        cond_prob = joint_dist.loc[curr_most_like_token["phone"], token_row.name]["p"] / token_row["p"]
        purity += token_row["p"] * cond_prob
    logger.info(f'Token purity: {purity:.3f}')
    return purity


def calc_NMI(joint_dist, marginal_phone_dist, marginal_token_dist):
    """
    phone-normalized mutual information
    """
    merged = joint_dist.reset_index()
    merged = merged.merge(marginal_phone_dist, on="phone", suffixes=('', '_phone'))
    merged = merged.merge(marginal_token_dist, on="token", suffixes=('', '_token'))

    assert joint_dist.index.is_unique, "Index is not unique"
    assert np.isclose(merged['p'].sum(), 1), f"Sum of joint distribution is not 1: {merged['p'].sum()}"
    assert (merged['p_phone'] <= 1).all(), "Marginal phone distribution has values greater than 1"
    assert (merged['p_token'] <= 1).all(), "Marginal token distribution has values greater than 1"
    assert (merged['p'] >= 0).all(), "Joint distribution has negative values"
    assert (merged['p_phone'] >= 0).all(), "Marginal phone distribution has negative values"
    assert (merged['p_token'] >= 0).all(), "Marginal token distribution has negative values"

    I = (merged['p'] * np.log2(merged['p'] / (merged['p_phone'] * merged['p_token']))).sum()
    PH = -(marginal_phone_dist['p'] * np.log2(marginal_phone_dist['p'])).sum()
    TH = -(marginal_token_dist['p'] * np.log2(marginal_token_dist['p'])).sum()
    PNMI = I / PH
    TNMI = I / TH
    logger.info(f'PNMI: {PNMI:.3f}, TNMI: {TNMI:.3f}')
    return PNMI, TNMI


def calc_joint_distribution(df: pd.DataFrame):
    joint_dist = df.groupby(["phone", 'token']).size().reset_index(name="count")
    joint_dist["p"] = joint_dist["count"] / joint_dist["count"].sum()
    joint_dist = joint_dist.set_index(["phone", 'token'])
    assert np.isclose(joint_dist["p"].sum(), 1), f"Sum of joint distribution is not 1: {joint_dist['p'].sum()}"
    assert (joint_dist["p"] >= 0).all(), "Joint distribution has negative values"
    assert (joint_dist["p"] <= 1).all(), "Joint distribution has values greater than 1"
    return joint_dist


def calc_marginal_phone_dist(joint_dist: pd.DataFrame):
    assert joint_dist.index.is_unique, "Index is not unique"
    marginal_dist = joint_dist.groupby("phone")['p'].sum().reset_index(name="p")
    marginal_dist = marginal_dist.set_index("phone")
    assert np.isclose(marginal_dist["p"].sum(), 1), f"Sum of marginal distribution is not 1: {marginal_dist['p'].sum()}"

    most_like_token_by_phone = joint_dist.reset_index().groupby("phone").apply(func=lambda x: x.loc[x["p"].idxmax()]).reset_index(drop=True)
    most_like_token_by_phone = most_like_token_by_phone.set_index("phone")
    return marginal_dist, most_like_token_by_phone


def calc_marginal_token_dist(joint_dist: pd.DataFrame):
    assert joint_dist.index.is_unique, "Index is not unique"
    marginal_dist = joint_dist.groupby('token')['p'].sum().reset_index(name="p")
    marginal_dist = marginal_dist.set_index("token")
    assert np.isclose(marginal_dist["p"].sum(), 1), f"Sum of marginal distribution is not 1: {marginal_dist['p'].sum()}"

    most_like_phone_by_token = joint_dist.reset_index().groupby("token").apply(func=lambda x: x.loc[x["p"].idxmax()]).reset_index(drop=True)
    most_like_phone_by_token = most_like_phone_by_token.set_index("token")
    return marginal_dist, most_like_phone_by_token


def calc_all_metrics(df: pd.DataFrame, json_output_path):
    df = df.copy()
    joint_dist = calc_joint_distribution(df)
    marginal_phone_dist, most_like_token_by_phone = calc_marginal_phone_dist(joint_dist)
    marginal_token_dist, most_like_phone_by_token = calc_marginal_token_dist(joint_dist)
    phone_purity = calc_phone_purity(joint_dist, marginal_token_dist, most_like_phone_by_token)
    token_purity = calc_token_purity(joint_dist, marginal_phone_dist, most_like_token_by_phone)
    PNMI, TNMI = calc_NMI(joint_dist, marginal_phone_dist, marginal_token_dist)
    results = {
        'PNMI': PNMI,
        'TNMI': TNMI,
        'phone_purity': phone_purity,
        'token_purity': token_purity,
    }
    json_output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {json_output_path}")
    return phone_purity, token_purity, PNMI, TNMI


def create_df(first_token: torch.Tensor, hop_length: int):
    data_dict = {f'token': first_token}
    data_dict['start_token'] = torch.arange(0, first_token.shape[0]) * hop_length
    data_dict['end_token'] = torch.arange(1, first_token.shape[0] + 1) * hop_length
    df = pd.DataFrame(data_dict)
    return df


def read_one_wav_and_phones(line: dict, model_sr) -> pd.DataFrame:
    wav, sr = torchaudio.load(line['path'])
    if wav.shape[0] == 2:
        wav = wav[:1]
    if sr != model_sr:
        wav = torchaudio.transforms.Resample(sr, model_sr)(wav)
    wav = wav.unsqueeze(0)
    phone_df = pd.read_csv(line['phones_path'], sep=' ', header=None, names=['start_phone', 'end_phone', 'phone'])
    return wav, phone_df


def encode_from_file(model, wav, sr):
    wav = wav.to(model.device)
    with torch.no_grad():
        encoded_frames, scale = model.encode(wav)
        encoded_tokens = encoded_frames.squeeze(0).cpu()
    hop_length = sr / model.frame_rate
    first_token = encoded_tokens[0]
    df = create_df(first_token, hop_length)
    return df


def merge_tokens_with_phones(tokens_df, phone_df):
    phone_df_extend = pd.concat(
        [
            pd.DataFrame({'start_phone': [0], 'end_phone': [phone_df['start_phone'].iloc[0]], 'phone': 'miss_start'}),
            phone_df,
            pd.DataFrame({'start_phone': [phone_df['end_phone'].iloc[-1]], 'end_phone': [tokens_df['end_token'].iloc[-1]], 'phone': 'miss_end'}),
        ]
    )
    merged_df = pd.merge(tokens_df, phone_df_extend, how='cross')
    merged_df['overlap'] = merged_df['end_phone'].clip(upper=merged_df['end_token']) - merged_df['start_phone'].clip(lower=merged_df['start_token'])
    final_df = merged_df.groupby('start_token', group_keys=False).apply(lambda x: x.loc[x['overlap'].idxmax()]).reset_index(drop=True)
    if (final_df['overlap'] <= 0).any():
        raise ValueError(f"Overlap is negative: {final_df['overlap']}")
    final_df['overlap_percentage'] = final_df['overlap'] / (final_df['end_token'] - final_df['start_token'])
    return final_df


def precit(model, manifest_path, csv_output_path: Path = None):
    if csv_output_path is not None and csv_output_path.exists():
        logger.info(f"CSV output path {csv_output_path} already exists. Skipping processing.")
        return pd.read_csv(csv_output_path)
    manifest_path = Path(manifest_path) if not isinstance(manifest_path, Path) else manifest_path

    dfs = []
    for line in tqdm(manifest_path.read_text().splitlines()):
        line_dict = json.loads(line)
        wav, phone_df = read_one_wav_and_phones(line_dict, model.sample_rate)
        tokens_df = encode_from_file(model, wav, model.sample_rate)
        merge_df = merge_tokens_with_phones(tokens_df, phone_df)
        dfs.append(merge_df)
    merged_df = pd.concat(dfs)
    logger.info(f"The data length is {len(merged_df)}")
    if csv_output_path:
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(csv_output_path, index=False)
        logger.info(f"Saved to {csv_output_path}")
    logger.info(merged_df['overlap_percentage'].describe())
    return merged_df


def main(model_cp, output_path, timit_manifest):
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_cp).stem if model_cp.endswith('.th') else model_cp
    model = PastModel.from_pretrained(model_cp)
    json_output_path = output_path / f"{model_name}_phones_eval.json"
    csv_output_path = output_path / f"{model_name}_phones_eval.csv"
    df = precit(model, timit_manifest, csv_output_path)

    calc_all_metrics(df, json_output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cp", type=str, help="Path or name of the model checkpoint")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--timit-manifest", type=Path, help="Path to the TIMIT manifest file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_logger(logger)
    main(**parse_args().__dict__)
