# README

Author: 102203558

## Overview
This repository contains a Python script demonstrating the TOPSIS method to select 
the best pre-trained model for sentence similarity tasks. Models include:

1. BERT-base
2. Sentence-BERT
3. Universal Sentence Encoder
4. MiniLM
5. GPT-embeddings

## Requirements
- Python 3.7+
- NumPy

Install NumPy using:
pip install numpy

## How to Run
1. Clone or download this repository.
2. Open a terminal and navigate to the folder containing the script.
3. Run:
   python topsis_sentence_similarity.py

4. The script displays each model’s closeness coefficient and the final ranking.

## File Description
- topsis_sentence_similarity.py:
  - Defines a decision matrix with criteria (accuracy, speed, model size).
  - Assigns weights and flags (beneficial or non-beneficial).
  - Implements the TOPSIS calculation and prints results.

## Notes
- You can replace the data in the decision matrix to reflect real measurements.
- Adjust weights or beneficial flags based on your priorities for different criteria.
