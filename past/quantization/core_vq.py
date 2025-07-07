import typing as tp

import torch
from audiocraft.quantization.core_vq import ResidualVectorQuantization as ResidualVectorQuantizationAudiocraft


class ResidualVectorQuantization(ResidualVectorQuantizationAudiocraft):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        all_centroids = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            all_centroids.append(residual + (quantized - residual).detach())
            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            quantized_out = x + (quantized_out - x).detach()

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        out_centroids = torch.stack(all_centroids, dim=1)
        return quantized_out, out_indices, out_losses, out_centroids
