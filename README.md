# Image Captioning Using Deep Learning

An image captioning application that generates natural-language descriptions of images using **EfficientNetV2B0, GRU, and Bahdanau Attention**. The project includes a Streamlit interface, model training and evaluation scripts, and attention visualization tools.

## Features

- Generate captions for uploaded images.
- Extract image features using EfficientNetV2B0.
- Generate text using a GRU-based decoder with Bahdanau Attention.
- Visualize attention during caption generation.
- Evaluate captions using BLEU, ROUGE-L, and CIDEr metrics.
- Explore model comparisons and dataset vocabulary through Streamlit pages.

## Model Architecture

1. **Image Encoder:** EfficientNetV2B0 extracts visual features.
2. **Attention Mechanism:** Bahdanau Attention helps the decoder focus on relevant image features.
3. **Text Decoder:** A GRU generates the caption token by token.
4. **Evaluation:** Generated captions are evaluated using standard captioning metrics.

## Project Structure

```text
Image-Captioning/
├── app/
│   ├── streamlit_app.py
│   ├── components.py
│   └── pages/
│       ├── 1_Caption_Image.py
│       ├── 2_Attention_Viewer.py
│       ├── 3_Model_Comparison.py
│       └── 4_Dataset_and_Vocab.py
├── scripts/
│   ├── download_data.py
│   └── make_samples.py
├── src/
│   ├── models/
│   ├── config.py
│   ├── dataset.py
│   ├── decode.py
│   ├── evaluate.py
│   ├── features.py
│   ├── inference.py
│   ├── text.py
│   └── train.py
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gauravkr25/Image-Captioning.git
cd Image-Captioning
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset and Model Setup

The Flickr8k dataset and trained model artifacts are not included in this repository.

Before running inference, prepare the dataset and model files according to the paths expected by the project configuration.

The trained model directory used in the development environment was:

```text
artifacts/models/flickr8k_gpu_full/
```

It contained model weights, a decoder, tokenizer vocabulary, and configuration files. These artifacts are excluded from Git because of their size.

**Important:** A fresh clone requires the compatible trained artifacts and any required dataset files before all application features can work. The exact download and setup process should be followed from the project's scripts and configuration.

## Run the Application

From the repository root, run:

```bash
python -m streamlit run app/streamlit_app.py
```

If Streamlit is installed but the application cannot import project modules, try:

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Evaluation Results

The following results were recorded during development on 809 evaluation images.

| Metric | Score |
|---|---:|
| BLEU-1 | 0.6031 |
| BLEU-2 | 0.4158 |
| BLEU-3 | 0.2799 |
| BLEU-4 | 0.1818 |
| ROUGE-L | 0.3782 |
| CIDEr | 0.4514 |

Results may vary depending on preprocessing, model checkpoints, decoding settings, and evaluation implementation.

## Testing

Run the test suite from the project root:

```bash
python -m pytest -q
```

Tests may require additional dependencies or fixtures, depending on the test configuration.

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetV2B0
- GRU
- Bahdanau Attention
- Streamlit
- NumPy
- Pillow
- pytest

## Future Improvements

- Improve caption quality through additional training and tuning.
- Compare greedy decoding with beam search.
- Improve attention visualization.
- Simplify model artifact setup for new users.
- Provide a hosted demo.

## Author

**Gaurav Kumar**

GitHub: [gauravkr25](https://github.com/gauravkr25)

---

*This project was developed for learning and experimentation in computer vision, deep learning, and natural-language generation.*
