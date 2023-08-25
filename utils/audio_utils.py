import matplotlib.pyplot as plt
import librosa

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", dB=False):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    if dB:
        im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    else:
        im = axs.imshow(specgram, origin="lower", aspect="auto") 
    fig.colorbar(im, ax=axs)
    plt.show(block=False)