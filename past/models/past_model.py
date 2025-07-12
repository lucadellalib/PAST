import logging
import os
import typing as tp
from pathlib import Path

import torch
from torch import nn

from past.models.encodec import EncodecModel

logger = logging.getLogger()


class PastModel(EncodecModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer,
        frame_rate: int,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        renormalize: bool = False,
        auxiliary_tasks_models: tp.List[nn.Module] = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            frame_rate=frame_rate,
            sample_rate=sample_rate,
            channels=channels,
            causal=causal,
            renormalize=renormalize,
        )
        self.auxiliary_tasks_models = auxiliary_tasks_models

    @classmethod
    def from_pretrained(cls, model_cp: str, device: tp.Union[torch.device, str] = None):
        """Instantiate a CompressionModel from a given checkpoint path or huggingface model name.
        This method is a convenient endpoint to load a CompressionModel to use in other solvers.

        Args:
            checkpoint (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
            device (torch.device or str): Device on which the model is loaded.
        """
        from past.models.builders import (get_compression_model,
                                          get_model_cp_from_huggingface)

        if device is None:
            device = "cpu"
        if not os.path.exists(model_cp):
            # If the path is not a file, try to get it from HuggingFace hub
            checkpoint_path = get_model_cp_from_huggingface(model_cp)
            logger.info(
                f"Checkpoint path downloaded from HuggingFace successfully: {checkpoint_path}"
            )
        else:
            checkpoint_path = Path(model_cp)
            logger.info(
                f"Loading local compression model from checkpoint: {checkpoint_path}"
            )
        assert (
            checkpoint_path is not None and Path(checkpoint_path).exists()
        ), f"Checkpoint not found: {checkpoint_path}"
        state = torch.load(checkpoint_path, map_location="cpu")
        assert (
            state is not None and "xp.cfg" in state
        ), f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state["xp.cfg"]
        cfg.device = device
        compression_model = get_compression_model(cfg).to(device)
        assert (
            compression_model.sample_rate == cfg.sample_rate
        ), "Compression model sample rate should match"
        assert "best_state" in state and state["best_state"] != {}
        state_dict_pruned = state["best_state"]["model"]
        for k in list(state_dict_pruned.keys()):
            if k.startswith("auxiliary_tasks_models."):
                del state_dict_pruned[k]
        compression_model.load_state_dict(state_dict_pruned)
        compression_model.eval()
        logger.info("Compression model loaded!")
        return compression_model

    @property
    def has_axiliary_task(self):
        return (
            self.auxiliary_tasks_models is not None
            and len(self.auxiliary_tasks_models) > 0
        )

    @property
    def device(self):
        """Return the device on which the model is loaded."""
        return next(self.parameters()).device

    @staticmethod
    def calc_frame_valid_mask(audio, audio_tokens, hop_length):
        valid_mask = torch.ones_like(
            audio_tokens[:, 0], dtype=torch.bool, device=audio_tokens.device
        )
        T_audio = audio.shape[2]
        T_tokens = audio_tokens.shape[2]
        last_non_zero_audio_index = T_audio - (
            torch.flip(audio, [-1]) != 0
        ).float().argmax(dim=-1)
        last_non_zero_tokens_index = torch.ceil(last_non_zero_audio_index / hop_length)
        padding_cond = last_non_zero_tokens_index < torch.arange(
            end=T_tokens, device=audio.device
        ).unsqueeze(0)
        valid_mask[padding_cond] = 0
        return valid_mask

    def forward(self, x: torch.Tensor, targets: dict = None):
        q_res = super().forward(x)
        if targets is None or len(targets) == 0:
            return q_res, torch.tensor(0.0, device=q_res.codes.device), {}
        valid_frame_mask = self.calc_frame_valid_mask(
            x, q_res.codes, self.encoder.hop_length
        )

        loss = torch.tensor(0.0, device=q_res.codes.device)
        metrics = {}
        if self.has_axiliary_task:
            semantic_error = torch.tensor(0.0, device=q_res.codes.device)
            for aux_model in self.auxiliary_tasks_models:
                loss_, metrics_ = aux_model(q_res, targets, valid_frame_mask)
                loss += loss_ * aux_model.weight
                semantic_error += metrics_.pop("semantic_error", 0.0)
                metrics.update(metrics_)
            metrics["semantic_error"] = semantic_error
        return q_res, loss, metrics

    def decode(
        self,
        codes: torch.Tensor,
        scale: tp.Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        if return_latent:
            return emb
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out


if __name__ == "__main__":
    model = PastModel.from_pretrained(
        "PAST_streamable"
    )  # one of ['PAST', 'PAST_streamable']
    wav = torch.randn(2, 1, 16000)
    codes = model.encode(wav)[0]
    reconstructed = model.decode(codes)
    print(model)
