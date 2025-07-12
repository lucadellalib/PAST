"""
All the functions to build the relevant models and modules
from the Hydra config.
"""

import logging
from pathlib import Path

import omegaconf

from past.models.past_model import PastModel
from past.modules.seanet import SEANetDecoder, SEANetEncoder
from past.quantization.vq import ResidualVectorQuantizer

logger = logging.getLogger()


def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    """Convenience function to map an omegaconf configuration to a dictionary.

    Args:
        cfg (omegaconf.DictConfig): Original configuration to map to dict.
    Returns:
        dict: Config as dictionary object.
    """
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct


def get_quantizer(quantizer: str, cfg: omegaconf.DictConfig, dimension: int):
    klass = {"rvq": ResidualVectorQuantizer}[quantizer]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != "no_quant":
        kwargs["dimension"] = dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name == "seanet":
        kwargs = dict_from_config(getattr(cfg, "seanet"))
        transformer_params = kwargs.pop("transformer_params", dict())
        encoder_override_kwargs = kwargs.pop("encoder")
        decoder_override_kwargs = kwargs.pop("decoder")
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        if transformer_params.pop("do_dec", False):
            decoder_kwargs["transformer_params"] = transformer_params
        if transformer_params.pop("do_enc", False):
            encoder_kwargs["transformer_params"] = transformer_params
        encoder = SEANetEncoder(**encoder_kwargs)
        decoder = SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_compression_model(cfg: omegaconf.DictConfig) -> PastModel:
    """Instantiate a compression model."""
    kwargs = dict_from_config(getattr(cfg, "encodec"))
    encoder_name = kwargs.pop("autoencoder")
    quantizer_name = kwargs.pop("quantizer")
    encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
    quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
    frame_rate = kwargs["sample_rate"] // encoder.hop_length
    renormalize = kwargs.pop("renormalize", False)
    # deprecated params
    kwargs.pop("renorm", None)
    return PastModel(
        encoder,
        decoder,
        quantizer,
        frame_rate=frame_rate,
        renormalize=renormalize,
        **kwargs,
    ).to(cfg.device)


def get_model_cp_from_huggingface(model_name: str) -> Path:
    """Get a model checkpoint from HuggingFace hub."""
    from huggingface_hub import hf_hub_download

    REPO_ID = "slprl/PAST"
    if not model_name.endswith(".th"):
        model_name += ".th"
    model_cp_path = hf_hub_download(repo_id=REPO_ID, filename=model_name)
    model_cp_path = Path(model_cp_path)
    logger.info(f"Model checkpoint downloaded to: {model_cp_path}")
    return model_cp_path
