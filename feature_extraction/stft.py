import torch
import torchaudio

class STFT:
    def __init__(self, n_fft=512, hop_length=256, win_length=512, window='hamming_window'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        
    def __call__(self, waveform):
        """ waveform: (B, samples)
            out: (B, T, F)
        """
        out = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(waveform.device),
            return_complex=True
        ).transpose(1, 2)
        return out
    
class ISTFT:
    def __init__(self, n_fft=512, hop_length=256, win_length=512, window='hammin_window'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        
    def __call__(self, stft_matrix):
        """ stft_matrix: (B, T, F)
            out: (B, samples)
        """
        out = torch.istft(
            stft_matrix.transpose(1, 2),
            self.n_fft,
            self.hop_length, 
            self.win_length,
            self.window.to(stft_matrix.device)
        )
        return out

if __name__ == "__main__":
    # Usage example
    n_fft = 400
    hop_length = 160
    win_length = 400

    stft = STFT(n_fft, hop_length, win_length)
    istft = ISTFT(n_fft, hop_length, win_length)

    waveform, sample_rate = torchaudio.load("path_to_audio.wav", normalize=True)
    stft_matrix = stft(waveform)
    reconstructed_waveform = istft(stft_matrix)

    # Save reconstructed audio
    torchaudio.save("reconstructed_audio.wav", reconstructed_waveform, sample_rate)
