# 🦠 TARP: Transformers for Antimicrobial Resistance Prediction 
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![GitHub License](https://img.shields.io/github/license/debugst1ck/TARP.svg)](https://github.com/debugst1ck/TARP/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/debugst1ck/TARP.svg)](https://github.com/debugst1ck/TARP/issues)
[![GitHub Stars](https://img.shields.io/github/stars/debugst1ck/TARP.svg)](https://github.com/debugst1ck/TARP/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/debugst1ck/TARP.svg)](https://github.com/debugst1ck/TARP/fork)

This repository is a suite of tools and models designed to predict antimicrobial resistance (AMR) using transformer-based architectures. The project uses state-of-the-art techniques in natural language processing (NLP) to analyze genetic sequences and predict resistance profiles.

## ✨ Features
- Implementation of transformer and traditional architectures tailored for AMR prediction.
- Data preprocessing pipelines for genetic sequences.
- Automatic mixed precision training for improved performance.
- Support for various datasets and easy integration of new data sources.

## 🚀 Getting Started
1. Clone the repository:
    ```bash
    git clone https://github.com/debugst1ck/TARP.git
    ```
2. Navigate to the project directory:
    ```bash
    cd TARP
    ```
3. (Optional, Recommended) Create and activate a virtual environment:
    For Windows PowerShell:
    ```powershell
    Set-ExecutionPolicy RemoteSigned -Scope Process
    python -m venv .venv
    .venv\Scripts\activate
    ```
    For Unix:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
4. Install the required dependencies (use the correct index URL for your CUDA version):
    ```bash
    pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128 # For CUDA 12.8
    ```
5. Prepare your dataset in the required format (FASTA files with corresponding labels).
6. Run the training script with your dataset:
    ```bash
    tarp
    ```

## 👨‍💻 Developer's notes
The codebase is structured to facilitate easy experimentation with different transformer architectures and hyperparameters. The main components include data preprocessing, model training, evaluation, and visualization of results.

### 🧠 Attention Mask
A value of 1 or True indicates that the model should attend to this position. This is for the actual content of the input. A value of 0 or False indicates that the model should not attend to this position, typically because it is padding.

### 🏷️ Class Weights
Class weights are calculated to address class imbalance in the dataset. The weights are inversely proportional to the frequency of each class, ensuring that the model pays more attention to minority classes during training.

$$
\text{weight}_i = \frac{N}{C \cdot n_i}
$$
