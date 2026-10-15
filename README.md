# Image Captioning Using Deep Learning

An AI-powered Image Captioning application that generates natural-language descriptions of images using deep learning and attention mechanisms.

## Overview

This project combines computer vision and natural language processing to generate captions for images. It uses an EfficientNetV2B0 encoder to extract image features and a GRU-based decoder with Bahdanau Attention to generate captions.

The project also includes a Streamlit web application for generating captions and visualizing attention.

## Features

- Automatic image caption generation
- EfficientNetV2B0 image feature extraction
- GRU-based decoder with Bahdanau Attention
- Attention visualization for generated captions
- Streamlit web interface
- Model evaluation using BLEU, ROUGE-L, and CIDEr metrics
- Dataset and vocabulary utilities

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Encoder:** EfficientNetV2B0
- **Caption Decoder:** GRU with Bahdanau Attention
- **Web Interface:** Streamlit
- **Dataset:** Flickr8k

## Model Architecture

1. **Image Encoder:** Extracts visual features using EfficientNetV2B0.
2. **Attention Mechanism:** Helps the decoder focus on relevant image features while generating each word.
3. **Text Decoder:** Uses a GRU network to generate a caption token by token.
4. **Output:** Produces a natural-language description of the input image.

## Evaluation

The trained model was evaluated on the Flickr8k test split.

| Metric | Score |
|---|---:|
| BLEU-1 | 0.6031 |
| BLEU-2 | 0.4158 |
| BLEU-3 | 0.2799 |
| BLEU-4 | 0.1818 |
| ROUGE-L | 0.3782 |
| CIDEr | 0.4514 |

These scores reflect the evaluation run performed for this project; results may vary with model configuration and decoding settings.

## Project Structure

```text
Image-Captioning/
├── app/              # Streamlit application
├── configs/          # Model and training configurations
├── data/             # Dataset files (images excluded from Git)
├── docs/             # Documentation
├── scripts/          # Utility scripts
├── src/              # Model, training, and inference code
├── tests/            # Automated tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gauravkr25/Image-Captioning.git
cd Image-Captioning
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

From the project root, run:

```bash
$env:PYTHONPATH="."
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Then open `http://localhost:8501` in your browser.

**Note:** Running the application requires the model files, vocabulary, and any required feature artifacts to be available locally. These generated files are excluded from Git, so configure or generate them according to the project scripts and configuration files.

## Dataset

This project uses the Flickr8k image-caption dataset. Dataset images are excluded from this repository because of their size and dataset distribution requirements. Obtain the dataset separately and place the required files in the expected project directories.

## Future Improvements

- Experiment with Transformer-based caption decoders
- Improve caption quality and evaluation scores
- Add support for more datasets
- Optimize inference speed and deployment

## Author

**Gaurav Kumar**

B.Tech — Computer Science and Engineering

GitHub: [@gauravkr25](https://github.com/gauravkr25)

---

*Developed as a deep learning project combining computer vision, natural language processing, and attention mechanisms.*
