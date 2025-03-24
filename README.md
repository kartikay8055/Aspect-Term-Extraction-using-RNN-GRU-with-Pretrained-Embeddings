# Aspect Term Extraction (ATE) using RNN & GRU

## 1. Objective
This project focuses on **Aspect Term Extraction (ATE)** using **RNN and GRU models** with **pretrained GloVe and fastText embeddings**. The goal is to extract aspect terms from sentences using **BIO encoding**.

## 2. Preprocessing
- **Tokenization:** Each sentence was tokenized into individual words.
- **BIO Encoding:** Applied **Begin (B), Intermediate (I), and Other (O)** tagging to label aspect terms.
- **Data Formatting:** Created a structured dataset with **sentences, tokens, BIO labels, and aspect terms**.

## 3. Vocabulary Construction
- Extracted all unique tokens from sentences and stored them in a **vocabulary list (vocab variable)**.

## 4. Model Training
We trained the following models using the preprocessed dataset:

| Model                | Embedding Dimension | Hidden Layers | Dropout |
|----------------------|--------------------|--------------|---------|
| **RNN + GloVe**      | 50                 | 64           | 0.4     |
| **RNN + fastText**   | 300                | 32           | 0.4     |
| **GRU + GloVe**      | 50                 | 64           | 0.4     |
| **GRU + fastText**   | 300                | 32           | 0.4     |

## 5. Training and Validation Loss Plots


![Screenshot 2025-03-24 183813](https://github.com/user-attachments/assets/e3734a63-66d9-40cd-8911-753647b5d4ad)
![Screenshot 2025-03-24 183825](https://github.com/user-attachments/assets/b1721fa7-89eb-4c3f-a643-0b9749f96a98)



## 6. Performance Comparison & Best Model

| Model                | Chunk-Level F1 Score | Tag-Level Accuracy |
|----------------------|---------------------|--------------------|
| **GRU + fastText**   | **24.77%**           | 18.94%            |
| **GRU + GloVe**      | 22.14%               | 17.72%            |
| **RNN + fastText**   | 19.82%               | 18.94%            |
| **RNN + GloVe**      | **17.53% (Lowest)**  | 18.94%            |

- **GRU + fastText** performed the best in **chunk-level F1-score (24.77%)**.
- **RNN + GloVe** had the lowest **chunk-level F1-score (17.53%)**.
- **Tag-level accuracy** remained relatively stable across models.


## 8. Acknowledgments
- **Pretrained Embeddings:** GloVe, fastText
- **Sequence Labeling Models:** RNN, GRU
- **Evaluation Metrics:** Chunk-level and Tag-level F1-score

