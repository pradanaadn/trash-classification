# Trash Classification

## Project Overview
This project aims to classify different types of trash using a deep learning model. The model is trained on the `garythung/trashnet` dataset and uses a custom architecture called `Trashmobilenet-v1`.

## Project Structure

```
.
├── checkpoints
│   └── best_model.pth
├── LICENSE
├── main.py
├── Makefile
├── notebook
│   └── experiment_trash_classification.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── callback
│   │   └── earlystopping.py
│   ├── config.py
│   ├── data
│   │   └── preprocessing.py
│   ├── metric
│   │   └── metric.py
│   └── models
│       ├── evaluation.py
│       ├── model.py
│       └── training.py
├── test.py
└── uv.lock

```



## Requirements
- Python 3.12 or higher
- See `requirements.txt` for a full list of dependencies

## Setup
1. Clone the repository:
    ```sh
    git clone git@github.com:pradanaadn/trash-detection.git
    cd trash-detection
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Login to Weights & Biases:
    ```sh
    wandb login <your_wandb_api_key>
    ```

5. Login to Hugging Face:
    ```sh
    huggingface-cli login --token <your_hf_token>
    ```

## Running the Project
To train the model, run:
```sh
python main.py