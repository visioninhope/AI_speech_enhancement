from dataset.dataset_loader import create_data_loaders
from models.model_architecture import YourModel
from training_loop.trainer import train_model
from utils.audio_utils import load_audio_data
from utils.metrics import calculate_metrics
import torch
import hydra

@hydra.main(config_path="config", config_name="training_config")
def main(cfg):
    # Set up data loaders
    train_loader, val_loader = create_data_loaders(cfg.data)

    # Initialize model
    model = YourModel(cfg.model)

    # Move model to device (CPU or GPU)
    device = torch.device(cfg.training.device)
    model.to(device)

    # Train the model
    train_model(model, train_loader, val_loader, cfg.training)

    # Load and preprocess test audio data
    test_audio = load_audio_data(cfg.data.test_audio_path)
    enhanced_audio = model(test_audio.to(device))
    
    # Calculate metrics on enhanced audio
    metrics = calculate_metrics(test_audio, enhanced_audio)

    print("Metrics on enhanced audio:")
    print(metrics)

if __name__ == "__main__":
    main()
