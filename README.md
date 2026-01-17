# LLM-Generated-Text_Detection
# Title: LLM-generated Text Detection: Enhancing Accuracy using DistilBERT & XLM-RoBERTa

# Description

This project focuses on detecting machine-generated text (MGT) using deep learning-based embeddings from DistilBERT and XLM-RoBERTa models. The embeddings are classified using XGBoost to achieve high accuracy in distinguishing AI-generated and human-written text.

# Dataset Information

We used three datasets for training and evaluation:

LLM-Detect AI Generated Text Dataset (Kaggle)

29,145 essays (14,508 human-written, 14,637 machine-generated)

Daigt-V4 Dataset (Kaggle)

UHAT dataset (Kaggle)

3600 samples of text (1800 human-written, 1800 machine-generated)

73,573 essays (37,370 human-written, 36,203 machine-generated)

Both datasets contain text samples written by various LLMs, including GPT, LLaMA, Mistral, Claude, PaLM, and Cohere.

# Code Information

The code is organized into separate scripts for each step:

# DistilBERT Embeddings Generation

DistilBERT Embeddings generation (Daigt-V4).ipynb

DistilBERT Embeddings generation (LLM-Detect AI generated Text).ipynb

# XLM-RoBERTa Embeddings Generation

XLM-RoBERTa Embeddings generation (Daigt-V4).ipynb

XLM-RoBERTa Embeddings generation (LLM-Detect AI generated Text).ipynb

# XLM-Roberta on urdu dataset

xlm-roberta-on-urdu-dataset.ipynb

# Classification using XGBoost

xg-boost-on-distilbert-embeddings-daigtv4-dataset.ipynb

xg-boost-on-distilbert-embeddings-llmdetectdataset.ipynb

xg-boost-on-xlm-embeddings-daigt-v4-dataset.ipynb

xg-boost-on-xlm-embeddings-llm-detect-dataset.ipynb

# Comparison with Machine learning models

logistic-regression-on-xlm-roberta-embeddings.ipynb

mlp-on-xlmroberta-embeddings.ipynb

random-forest-on-xlmroberta-embeddings.ipynb

# Usage Instructions

# Step 1: Install Dependencies

Ensure you have the required dependencies installed:

  pip install transformers torch xgboost pandas numpy scikit-learn
# Step 2: Run Embedding Generation

Execute the embedding generation notebooks to extract text embeddings:

  jupyter notebook "DistilBERT Embeddings generation (Daigt-V4).ipynb"
  
  jupyter notebook "DistilBERT Embeddings generation (LLM-Detect AI generated Text).ipynb"
  
  jupyter notebook "XLM-RoBERTa Embeddings generation (Daigt-V4).ipynb"
  
  jupyter notebook "XLM-RoBERTa Embeddings generation (LLM-Detect AI generated Text).ipynb"
Generated embeddings will be saved as CSV files on your Desktop.

# Step 3: Run Classification

Once embeddings are generated, run the classification model:

  jupyter notebook "xg-boost-on-distilbert-embeddings-daigtv4-dataset.ipynb"
  
  jupyter notebook "xg-boost-on-distilbert-embeddings-llmdetectdataset.ipynb"
  
  jupyter notebook "xg-boost-on-xlm-embeddings-daigt-v4-dataset.ipynb"
  
  jupyter notebook "xg-boost-on-xlm-embeddings-llm-detect-dataset.ipynb"
This will load the embeddings and train the XGBoost model for detecting AI-generated text.

# Methodology

 Preprocessing

Stopword removal

Lemmatization

Lowercasing

Punctuation & whitespace removal

Tokenization (using pre-trained tokenizers)

# Embedding Extraction

DistilBERT and XLM-RoBERTa extract deep semantic features

# Classification with XGBoost

Uses extracted embeddings to classify text

# Evaluation Metrics

Accuracy, precision, recall, and F1-score

# Requirements

Python 3.8+

PyTorch

Transformers (Hugging Face)

XGBoost

Pandas & NumPy

Scikit-learn

# Citations

LLM-Detect AI Generated Text Dataset (Kaggle): Kaggle Dataset: (https://www.kaggle.com/datasets/thedrcat/daigt-v4-train-dataset)

Daigt-V4 Dataset (Kaggle): Kaggle Dataset: Daigt-V4 Dataset

UHAT Dataset (Kaggle): Kaggle Dataset:[https://www.kaggle.com/datasets/ammarshafiq/urdu-human-and-ai-text-dataset-uhat]

DistilBERT: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. URL: https://arxiv.org/abs/1910.01108

XLM-RoBERTa: Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. URL: https://arxiv.org/abs/1911.02116

XGBoost: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. URL: https://arxiv.org/abs/1603.02754

Transformers Library (Hugging Face): Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art Natural Language Processing. URL: https://arxiv.org/abs/1910.03771

PyTorch: Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. URL: https://arxiv.org/abs/1912.01703

Scikit-learn: Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830. URL: http://jmlr.csail.mit.edu/papers/volume12/pedregosa11a/pedregosa11a.pdf
