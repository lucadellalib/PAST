import argparse
import logging
import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import torch
import torch.multiprocessing as mp
from torchaudio.functional import forced_align
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audiocraft.data.audio import audio_info, audio_read
from past.data.audio_dataset import AudioMeta

from utils import set_logger, get_device

logger = logging.getLogger(__name__)

DEFAULT_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
HOP_LEN = 320


def one_dataset(args, root, name):
    """
    Process a dataset by scanning audio files, aligning transcriptions (optional), and saving metadata.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        root (str): Root directory of the dataset.
        name (str): Name of the dataset (used for output and logic).

    Returns:
        dict: Mapping from split names to their corresponding .jsonl file paths.
    """
    run_transcriptions_alignment = not args.skip_transcriptions_alignment
    output_dir = Path(args.output_dir)
    logger.info(f"Scanning {root} for audio files...")
    all_files = scan_files(root, debug=args.debug)

    transcriptions_dir = (output_dir / 'transcriptions_alignment') if run_transcriptions_alignment else output_dir
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    tmp_paths = []
    if args.num_processes == 1:
        logger.info(f"Processing {name} dataset with a single process...")
        tmp_paths.append(f'/tmp/{name}_0_{os.getpid()}.jsonl')
        process_one_batch(all_files, tmp_paths[0], name, 0, transcriptions_dir, run_transcriptions_alignment)
    else:
        run_in_parallel(all_files, name, tmp_paths, args.num_processes, transcriptions_dir, run_transcriptions_alignment)

    logger.info(f"Combining temporary files into {name} dataset...")
    train_meta, dev_meta, test_meta = collect_and_split_metas(tmp_paths)

    logger.info(f"Saving {name} dataset...")
    save_manifests(train_meta, dev_meta, test_meta, output_dir, name)


def scan_files(root, debug=False):
    """Recursively scan folder for supported audio files."""
    all_files = []
    for root, _, files in os.walk(root, followlinks=True):
        for file in files:
            full_path = Path(root) / file
            if full_path.suffix in DEFAULT_EXTS:
                all_files.append(full_path)
            if debug and len(all_files) >= 200:
                logger.info(f"Debug mode: limiting to 200 files. Found {len(all_files)} files so far.")
                return all_files
    logger.info(f"Found {len(all_files)} audio files.")
    return all_files


def run_in_parallel(all_files, name, tmp_paths, num_processes, transcriptions_dir, run_transcriptions_alignment):
    """Process audio files in parallel using multiprocessing."""
    paths = [all_files[i::num_processes] for i in range(num_processes)]
    assert sum(len(p) for p in paths) == len(all_files)
    mp.set_start_method('spawn', force=True)
    processes = []
    for i, paths_batch in enumerate(paths):
        tmp_path = f'/tmp/{name}_{i}_{os.getpid()}.jsonl'
        tmp_paths.append(tmp_path)
        p = mp.Process(target=process_one_batch, args=(paths_batch, tmp_path, name, i, transcriptions_dir, run_transcriptions_alignment))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Process {p.name} failed with exit code {p.exitcode}")


def process_one_batch(paths_batch, tmp_path, name, i, transcriptions_dir, run_transcriptions_alignment=True):
    """Process a batch of audio files and save metadata to a temporary file."""
    processor, model = None, None
    if run_transcriptions_alignment:
        device = get_device(i)
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    with open(tmp_path, 'w') as f:
        for path in tqdm(paths_batch):
            line = one_file_to_jsonl(path, processor, model, name, transcriptions_dir)
            f.write(json.dumps(line.__dict__) + '\n')


def one_file_to_jsonl(file_path, processor, model, name, transcriptions_dir):
    """Convert an audio file to AudioMeta including optional transcription alignment."""
    info = audio_info(file_path)
    amplitude = trans_path = None

    if processor and model:
        wav, sr = audio_read(file_path)
        amplitude = wav.abs().max().item()
        trans_path = (Path(transcriptions_dir) / file_path.parts[-3] / file_path.parts[-2] / file_path.name).with_suffix('.trans')
        if not trans_path.exists():
            transcription = get_transcription_timit(file_path) if 'timit' in name else get_transcription(file_path)
            alignment = calc_transcriptions_alignment(wav, sr, processor, model, transcription)
            save_transcriptions_alignment(trans_path, alignment)
        trans_path = str(trans_path)
    meta = AudioMeta(str(file_path), info.duration, info.sample_rate, amplitude, transcriptions_alignment_path=trans_path)
    if 'timit' in name:
        meta.phones_path = str(timit_wav_to_suffix(file_path))
    return meta


def calc_transcriptions_alignment(wav, sr, processor, model, transcription):
    """Run forced alignment of transcription to waveform."""
    wav = wav.reshape(-1)
    inputs = processor(wav, return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    targets = processor.tokenizer(transcription, return_tensors="pt").input_ids.to(model.device)
    paths, _ = forced_align(log_probs, targets)
    results = processor.decode(paths[0], output_char_offsets=True).char_offsets
    results_list = [(line['char'], line['start_offset'] * HOP_LEN, line['end_offset'] * HOP_LEN) for line in results]
    return results_list


def save_transcriptions_alignment(trans_path, alignment):
    """Save alignment output to CSV format."""
    df = pd.DataFrame(alignment, columns=['char', 'start_offset', 'end_offset'])
    trans_path.parent.mkdir(parents=True, exist_ok=True)
    trans_path.write_text(df.to_csv(index=False, sep=',', header=False))


def collect_and_split_metas(tmp_paths):
    """Load and categorize metadata by dataset split."""
    train_meta, dev_meta, test_meta = [], [], []
    for path in tmp_paths:
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                p = item['path'].lower()
                if 'dev' in p:
                    dev_meta.append(item)
                elif 'test' in p:
                    test_meta.append(item)
                elif 'train' in p:
                    train_meta.append(item)
                else:
                    raise ValueError(f"Unknown split in path: {p}")
        os.remove(path)
    return train_meta, dev_meta, test_meta


def save_manifests(train_meta, dev_meta, test_meta, output_dir, name):
    """Save dataset split metadata to JSONL files."""
    for split, meta in [('dev', dev_meta), ('test', test_meta), ('train', train_meta)]:
        if not meta:
            logger.warning(f"Empty {name} {split} dataset. Skipping...")
            continue
        path = output_dir / f'{name}_{split}.jsonl'
        with open(path, 'w') as f:
            for line in meta:
                f.write(json.dumps(line) + '\n')
        logger.info(f"Saved {name} {split} dataset with {len(meta)} samples to {path}")


def get_transcription(file):
    """Extract transcription for LibriSpeech-style filenames."""
    base = Path(file).parent
    name = Path(file).stem
    d1, d2, num = name.split('-')
    line = (base / f'{d1}-{d2}.trans.txt').read_text().splitlines()[int(num)]
    return line.split(' ', 1)[1]


def get_transcription_timit(wav_file):
    """Extract transcription for TIMIT file."""
    txt_file = timit_wav_to_suffix(wav_file, '.TXT')
    line = txt_file.read_text().strip().split(' ', 2)[-1].upper()
    return re.sub(r'[.,?\'":;]', '', line)


def timit_wav_to_suffix(wav_file, suffix='.PHN'):
    """Convert TIMIT WAV filename to another format (e.g., .PHN or .TXT)."""
    new_path = Path(re.sub(r'\.WAV\.wav$', suffix, str(wav_file)))
    assert new_path.exists(), f"File not found: {new_path}"
    return new_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate .jsonl metadata files from audio datasets.')
    parser.add_argument('--LibriSpeech_root', help='Root folder of the LibriSpeech dataset')
    parser.add_argument('--timit_root', help='Root folder of the TIMIT dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--skip_transcriptions_alignment', action='store_true', help='Skip transcription alignment step')
    parser.add_argument('--debug', action='store_true', help='Use only 200 files per dataset')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of parallel processes')
    return parser.parse_args()


def main(args):
    """Main entry point. Run processing on TIMIT and/or LibriSpeech datasets."""
    set_logger(logger)
    if args.skip_transcriptions_alignment:
        logger.warning("Skipping transcriptions alignment. You can't train auxiliary tasks without it.")
    if args.timit_root:
        one_dataset(args, args.timit_root, name='timit')
    if args.LibriSpeech_root:
        one_dataset(args, args.LibriSpeech_root, name='LibriSpeech')
    logger.info("All dataset processing complete.")


if __name__ == '__main__':
    main(parse_args())
