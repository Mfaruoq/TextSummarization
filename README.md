# Text Summarization using LSTM with Attention
This repository contains Python code for text summarization using LSTM (Long Short-Term Memory) with attention mechanism. The goal is to summarize lengthy news articles into concise summaries using a sequence-to-sequence (seq2seq) model.

# Dataset
The code uses two datasets: news_summary.csv and news_summary_more.csv. The former contains the main text, and the latter contains additional data for the summary generation.

# Requirements
To run the code, ensure you have the following libraries installed:

NumPy
Pandas
Scikit-learn
Keras
TensorFlow
spaCy
Matplotlib

# Getting Started
Clone this repository to your local machine.
Ensure the required libraries are installed by running:
Copy code
pip install numpy pandas scikit-learn keras tensorflow spacy matplotlib
Make sure you have the necessary data files (news_summary.csv and news_summary_more.csv) in the /kaggle/input directory.

# Code Overview
data_preprocessing.ipynb: This notebook contains data cleansing and preprocessing steps, including text cleaning and tokenization.
seq2seq_model.ipynb: This notebook builds the LSTM with attention-based sequence-to-sequence model for text summarization.
# How to Use
Run the text-summarization-with-seq2seq-model.ipynb notebook to perform data cleansing and preprocessing and train the seq2seq model for text summarization.
# Note
This code is intended for educational purposes and may require modifications for production use. Feel free to experiment with different model architectures, hyperparameters, and datasets to achieve better results.

# Credits
The code in this repository is based on previous work by various authors in the field of text summarization. The data used is sourced from Kaggle, and specific dataset references are provided within the code.

Please cite the relevant sources if you use this code or dataset for research or academic purposes.

# Happy Summarizing!
