import os
import json
import torch
import torchaudio
from tqdm import tqdm


class Metadata:
    def __init__(
            self,
            data
        ):
        self.data = data
        self.paths = [entry for entry in self.data if "path" in entry]

    def read_metadata(self, path):
        with open(os.path.join(path, "metadata.json"), "r") as metadata_file:
            return json.load(metadata_file)

    def generate_metadata(self):
        out = {}
        for p in self.paths:
            out[p] = []
            for i in range(len(self.data[p])): # It can be multiple subpaths
                if not os.path.isfile(os.path.join(self.data[p][i], "metadata.json")) and len(self.data[p][i]) > 0: # Generate metadata if file does not exist
                    metadata = []
                    for root, dirs, files in tqdm(os.walk(self.data[p][i]), total=len(list(os.walk(self.data[p][i]))), desc=f"Metadata-{p}-{i+1}"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            info = torchaudio.info(file_path)

                            metadata.append({
                                "file_name": file,
                                "file_path": file_path,
                                "duration_seconds": info.num_frames / info.sample_rate,
                                "num_channels": info.num_channels,
                                "sample_rate": info.sample_rate
                            })

                    with open(os.path.join(self.data[p][i], "metadata.json"), "w") as json_file:
                        json.dump(metadata, json_file, indent=4)  
                else: # Read metadata
                    metadata = self.read_metadata(self.data[p][i])
                out[p].append(metadata)
        return out