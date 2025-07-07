from dataclasses import dataclass
from pathlib import Path

import typing as tp
import pandas as pd

import torch
import numpy as np
from past.data.audio_dataset import AudioMeta, load_audio_meta
from past.data.phone_utils import phones2idx
import logging
from past.data.audio_dataset import AudioDataset

logger = logging.getLogger(__name__)


@dataclass
class PhoneSpeechDataset(AudioDataset):

    def __init__(self, meta: tp.List[AudioMeta], augmentin_flip_phase: bool = False, **kwargs):
        kwargs['return_info'] = True
        self.augmentin_flip_phase = augmentin_flip_phase
        self.do_phone = kwargs.pop('do_phone', False)
        self.do_char = kwargs.pop('do_char', False)
        self.hop_length = None
        super().__init__(meta, **kwargs)

    @classmethod
    def from_meta(cls, librispeech_path, timit_path, timit_probability, **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        max_duration = kwargs.pop('max_duration', None)

        lists = []
        weights = []

        def one_dataset(dataset_root: Path, weight: float):
            if weight == 0.0 or dataset_root is None:
                return
            curr_list = cls._read_one_manifest(Path(dataset_root))
            if max_duration is not None:
                curr_list = [m for m in curr_list if m.duration <= max_duration]
            lists.append(curr_list)
            weights.append(weight)

        one_dataset(librispeech_path, 1.0 - timit_probability)
        one_dataset(timit_path, timit_probability)

        kwargs['sample_on_weight'] = True
        if kwargs.get('sample_on_duration', False):
            updated_weights = [weight / sum([m.duration for m in curr_list]) for curr_list, weight in zip(lists, weights)]
        else:
            updated_weights = [weight / len(curr_list) for curr_list, weight in zip(lists, weights)]
        kwargs['sample_on_duration'] = False  # we insertthe duration in the weights

        meta = []
        for curr_list, curr_weight in zip(lists, updated_weights):
            for curr_meta in curr_list:
                curr_meta.weight = curr_weight
                meta.append(curr_meta)
        return cls(meta, **kwargs)

    @staticmethod
    def _read_one_manifest(root: Path):
        root = Path(root)
        if root.is_dir():
            if (root / 'data.jsonl').exists():
                root = root / 'data.jsonl'
            elif (root / 'data.jsonl.gz').exists():
                root = root / 'data.jsonl.gz'
            else:
                raise ValueError("Don't know where to read metadata from in the dir. " "Expecting either a data.jsonl or data.jsonl.gz file but none found.")
        meta = load_audio_meta(root)
        return meta

    def set_hop_length(self, hop_length: int):
        self.hop_length = hop_length

    def __getitem__(self, index: int):
        assert self.hop_length is not None, "Hop length must be set before using the dataset"
        wav, meta = super().__getitem__(index)

        if self.augmentin_flip_phase and np.random.rand() > 0.5:
            wav = -wav

        wav = wav.squeeze(0)  # Remove channel dimension
        wav_path = Path(meta.meta.path)

        if self.do_phone:
            if meta.meta.phones_path is None:
                phones_mat = torch.zeros((0, len(phones2idx)))  # Empty tensor
            else:
                start_sample = int(meta.seek_time * meta.sample_rate) if self.segment_duration else 0
                end_sample = start_sample + int(self.segment_duration * meta.sample_rate) if self.segment_duration else len(wav)
                phones_mat = self._creat_phone_mat(meta.meta.phones, start_sample, end_sample, idx_map=phones2idx)
        else:
            phones_mat = None

        if self.do_char:
            assert meta.meta.transcriptions_alignment_path is not None, f"Missing transcription for {wav_path}"
            start_sample = int(meta.seek_time * meta.sample_rate) if self.segment_duration else 0
            end_sample = start_sample + int(self.segment_duration * meta.sample_rate) if self.segment_duration else len(wav)
            transcription = self._load_transcription(meta.meta.transcriptions_alignment, start_sample, end_sample)
        else:
            transcription = None

        return wav, transcription, phones_mat

    def _creat_phone_mat(self, phone_list: list, start_sample: int, end_sample: int, idx_map):
        phones_start = np.array([0] + [s for p, s, e in phone_list] + [phone_list[-1][2]])
        phones_end = np.array([phone_list[0][1]] + [e for p, s, e in phone_list] + [end_sample])
        phones = np.array(['miss_start'] + [p for p, s, e in phone_list] + ['miss_end'])

        # Create a 2D array where rows represent windows and columns represent phones
        windows_start = np.arange(start_sample, end_sample, self.hop_length)[:, None]
        windows_end = np.arange(start_sample + self.hop_length, end_sample + self.hop_length, self.hop_length)[:, None]

        # Compute overlap between windows and phone intervals
        start_crop = np.maximum(windows_start, phones_start)  # Clipped start
        end_crop = np.minimum(windows_end, phones_end)  # Clipped end
        overlap = np.maximum(end_crop - start_crop, 0)  # Overlap lengths

        # Normalize overlaps to relative proportions within each window
        overlap_sum = overlap.sum(axis=1, keepdims=True)
        relavant_part = overlap / np.maximum(overlap_sum, 1e-6)

        # Map phone names to their indices using idx_map
        phone_indices = np.vectorize(idx_map.get)(phones)

        # Create the phones matrix using advanced indexing
        phones_mat = np.zeros((len(windows_start), len(idx_map)), dtype=np.float32)
        # Use a nested loop replacement for proper broadcasting
        for i in range(phone_indices.shape[0]):  # Iterate over windows
            phones_mat[:, phone_indices[i]] += relavant_part[:, i]

        # Convert the result to a PyTorch tensor
        phones_mat = torch.tensor(phones_mat).float()

        if '<pad>' in idx_map:
            phones_mat[phones_mat.sum(1) == 0, idx_map['<pad>']] = 1.0
        assert phones_mat.sum() > 0, f"Empty phone matrix"
        return phones_mat

    def _load_transcription(self, transcriptions_alignment: list, start_sample: int, end_sample: int):
        chars_in_range = [c for c, s, e in transcriptions_alignment if s >= start_sample and e <= end_sample]
        in_range_transcription = ''.join(chars_in_range).strip()
        return in_range_transcription

    def collater(self, samples):
        wavs, transcriptions, phones = zip(*samples)
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(dim=1)

        targets = {}
        if phones[0] is not None:
            targets['phone_mat'] = torch.nn.utils.rnn.pad_sequence(phones, batch_first=True)
        if transcriptions[0] is not None:
            targets['transcription'] = transcriptions

        return wavs, targets
