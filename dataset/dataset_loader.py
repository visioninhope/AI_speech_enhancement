import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

class SpeechEnhancementDataset(Dataset):
    def __init__(
            self,
            data,
            sample_rate=16000
        ):
        self.data = data
        self.sample_rate = sample_rate

        # Load metadata
        self.get_metadata()

    def get_metadata(self):
        with open(os.path.join(self.data.clean_path, "metadata.json"), "r") as metadata_file:
            self.clean_metadata = json.load(metadata_file)
        if not self.data.mixed_path:
            with open(os.path.join(self.data.noise_path, "metadata.json"), "r") as metadata_file:
                self.noise_metadata = json.load(metadata_file)
        else:
            with open(os.path.join(self.data.mixed_path, "metadata.json"), "r") as metadata_file:
                self.mixed_metadata = json.load(metadata_file)

    def read_audio(self, file_path, offset, frames):
        waveform, sample_rate = torchaudio.load(file_path, normalize=True, frame_offset=offset,num_frames=frames)
        if sample_rate != self.sample_rate:
            transform = T.Resample(sample_rate, self.sample_rate)
            waveform = transform(waveform)
        return waveform

    def __len__(self):
        return len(self.clean_metadata) # TODO: lenght according number specified in config file

    def __getitem__(self, idx):
        info = self.clean_metadata[idx]
        file_path = info["file_path"]

        # Load and normalize audio signal
        #torch.manual_seed(7) : TODO
        offset = 1*self.sample_rate # Dummy data for now
        frames = 3*self.sample_rate
        waveform = self.read_audio(file_path, offset, frames)

        # Implement clean signal generation and mixing here if needed TODO
        # ...

        return waveform, waveform

def create_data_loaders(data_config):
    train_data = SpeechEnhancementDataset(data=data_config.train, sample_rate=data_config.sample_rate)
    val_data = SpeechEnhancementDataset(data=data_config.val, sample_rate=data_config.sample_rate)

    train_loader = DataLoader(train_data, batch_size=data_config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=data_config.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
