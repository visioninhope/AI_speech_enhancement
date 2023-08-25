import torch
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
from feature_extraction.stft import STFT
from utils.audio_utils import plot_spectrogram


class FeatPipeline(torch.nn.Module):
    def __init__(
        self,
        # n_fft=512,
        # n_mel=256,
        # stretch_factor=0.8,
    ):
        super().__init__()

        self.spec = STFT(n_fft=512, hop_length=256, win_length=512)

        # self.spec_aug = torch.nn.Sequential(
        #     TimeStretch(stretch_factor, fixed_rate=True),
        #     FrequencyMasking(freq_mask_param=80),
        #     TimeMasking(time_mask_param=80),
        # )

        # self.mel_scale = MelScale(
        #     n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to STFT
        spec = self.spec(waveform)

        # Apply SpecAugment
        #spec = self.spec_aug(spec)

        # Convert to mel-scale
        # mel = self.mel_scale(spec)

        return spec