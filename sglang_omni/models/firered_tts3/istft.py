# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""iSTFT head with "same" padding and its streaming form.

The non-streaming ISTFT.forward / ISTFTHead.forward are ported from the
FireRedTTS3 reference implementation (fireredtts3/redae/redae.py);
forward_stream is the chunked equivalent (RedAE streaming decoder, verified
sample-exact against the full-segment path). Everything here is
precision-critical: the mag/phase construction, the OLA fold and the window
normalization must stay fp32 — bf16 phase accumulates audible distortion
across frames.
"""

import torch
import torch.nn as nn

# Lower bound for the OLA window envelope before normalization. The healthy
# envelope is COLA-flat (~0.75 at hop = win/4) in every region that survives
# trimming, so the clamp never touches legitimate samples; it only keeps a
# degenerate edge position (the streaming path skips the NOLA assert because
# chunk boundaries can legitimately approach zero there) from turning the
# fold into inf/NaN that would escape into the PCM cast on a trim slip.
ENVELOPE_FLOOR = 1e-8


class ISTFT(nn.Module):
    """Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than
    center=True) with windowing. This is because the NOLA (Nonzero Overlap Add) check fails
    at the edges. See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous
    to CNNs. The NOLA constraint is met as we trim padded samples anyway.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ):
        super().__init__()
        assert padding in ["center", "same"], "Padding must be 'center' or 'same'."
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the
                    output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y

    def forward_stream(
        self, spec: torch.Tensor, cache: torch.Tensor = None, last_chunk: bool = False
    ):
        """Forward only one frame.

        Args:
            spec: shape (B, N, T=chunk_size)
            cache: previous chunk's last ifft frame, shape (B, N, T=3)
            last_chunk: if last_chunk, will not trim the last (win-hop) segment
        Returns:
            y: shape (B, T=effective_length)
        """
        assert self.padding == "same", "Padding must be same."
        # The reference streaming implementation deliberately skips the NOLA
        # assert here: chunk boundaries carry the (win/hop - 1)-frame lookahead
        # tail whose envelope can legitimately approach zero; the padded
        # samples are trimmed anyway.
        assert (
            self.win_length % self.hop_length == 0
        ), f"{self.win_length} {self.hop_length}"
        pad = (self.win_length - self.hop_length) // 2

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]  # (B, N, T=chunk_size)

        # Append previous cache
        if cache is not None:
            ifft = torch.cat([cache, ifft], dim=-1)
        new_cache_t = self.win_length // self.hop_length - 1
        new_cache = ifft[..., -new_cache_t:]

        # Overlap and Add
        output_size = (ifft.shape[-1] - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        # Window envelope
        window_sq = (
            self.window.square().expand(1, ifft.shape[-1], -1).transpose(1, 2)
        )  # (B=1, N, T)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()

        # Normalize (envelope clamped: the streaming fold's padded edges may
        # legitimately sit at a near-zero envelope the trim discards).
        y = y / window_envelope.clamp_min(ENVELOPE_FLOOR)

        # Only take effective part
        if cache is None:
            y = y[:, pad:]
        else:
            y = y[:, (self.win_length - self.hop_length) :]
        if last_chunk:
            y = y[:, :-pad]
        else:
            y = y[:, : -(self.win_length - self.hop_length)]
        return y, new_cache

    def flush_stream(self, cache: torch.Tensor) -> torch.Tensor:
        """Terminal OLA tail from the cached windowed frames, no new input.

        The trailing (win_length - hop_length) / 2 samples of a stream depend
        only on the cached (already windowed) ifft frames, so the flush skips
        the irfft entirely — cuFFT rejects a zero-frame transform anyway.
        """
        pad = (self.win_length - self.hop_length) // 2
        ifft = cache
        output_size = (ifft.shape[-1] - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]
        window_sq = self.window.square().expand(1, ifft.shape[-1], -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()
        y = y / window_envelope.clamp_min(ENVELOPE_FLOOR)
        y = y[:, (self.win_length - self.hop_length) :]
        y = y[:, :-pad]
        return y


class ISTFTHead(nn.Module):
    """ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super().__init__()
        self.hop_length = hop_length
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of
                    the output signal.
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard against excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # directly produce the complex value (recomputing phase via atan2
        # would only cost time)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio

    def forward_stream(
        self, x: torch.Tensor, cache: torch.Tensor = None, last_chunk: bool = False
    ):
        """ISTFTHead can be adapted in streaming inference without retraining.

        Args:
            x: shape (B, T, C)
            cache: shape (B, N, T=3), istft cache
        Returns:
            audio: shape (B, t)
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)  # (B, C, T)
        mag = torch.clip(mag, max=1e2)  # safeguard against excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)  # (B, C, T)
        audio, new_cache = self.istft.forward_stream(S, cache, last_chunk)
        return audio, new_cache
