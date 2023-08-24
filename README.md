# AI_speech_enhancement
![SpeechEnhacement](https://github.com/eCabral87/AI_speech_enhancement/assets/19577030/3d55cc60-2fbe-4581-8e8c-3dcead806a63)

## Description

This repository contains the implementation of a speech enhancement project using deep learning techniques. The project aims to enhance the quality of noisy speech by applying advanced machine/deep learning models. It includes various components such as data preprocessing, model architecture, training loop, testing, and more.

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Modular folder structure for organized project management.
- Config files for flexible parameter settings in different project aspects.
- Data preprocessing, including raw audio processing and feature extraction.
- Data augmentation techniques for improving model generalization.
- Implementation of various deep learning models for speech enhancement.
- Training loop with support for knowledge distillation and quantization.
- Testing and evaluation using objective audio quality metrics.
- Easy-to-use scripts for training and testing.

## Folder Structure

The project follows a modular folder structure for organized development:

- `config`: Configuration files for different project aspects.
- `data`: Raw and processed audio data, along with metadata.
- `models`: Model architectures for speech enhancement.
- `features_extraction`: Feature extraction utilities.
- `data_augmentation`: Data augmentation techniques.
- `utils`: Utility functions for audio processing and metrics.
- `training_loop`: Training and validation loop.
- `testing`: Objective score calculation for testing.
- `dataset`: Data loading pipeline.
- `knowledge_distillation`: Implementation of knowledge distillation.
- `quantization`: Utilities for model quantization.

## Getting Started

To get started with the project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Configure your project settings in the respective configuration files.
4. Prepare your audio data and place it in the appropriate data folders.
5. Run the training script using `python main_train.py`.
6. Evaluate the model's performance using the provided testing scripts.

## Usage

- Modify the configuration files in the `config` folder to adjust various project settings.
- Place your raw audio data in the `data/raw_audio` directory and preprocess it using the provided tools.
- Customize the model architecture in the `models` folder according to your needs.
- Use data augmentation techniques from the `data_augmentation` folder to enhance your training data.
- Run the training loop using the `main_train.py` script.
- Evaluate your model's performance using the provided testing scripts in the `testing` folder.

## Configuration

The project uses configuration files to manage various settings:

- `data_config.yaml`: Data-related settings.
- `model_config.yaml`: Model architecture and hyperparameters.
- `training_config.yaml`: Training process settings.
- `distillation_config.yaml`: Knowledge distillation settings.
- `quantization_config.yaml`: Model quantization settings.

## Results

Describe the results of your trained models and their performance on your dataset.

## Contributing

Contributions to the project are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
