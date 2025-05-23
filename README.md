
# 📚 Multi-Label Text Classification for Academic Papers

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-yellow)
![Transformers](https://img.shields.io/badge/Sentence%20Transformers-2.2+-brightgreen)

This project implements a multi-label classification system to categorize academic papers into three domains: **Information Theory**, **Computational Linguistics**, and **Computer Vision**.

## 🚀 Key Features

- **Advanced Text Preprocessing**
  - Custom cleaning pipeline for academic text
  - Stopword removal and stemming
  - Handling of special characters and formatting

- **Multi-Label Classification**
  - Comparative analysis of KNN, Random Forest, and LightGBM
  - Sentence Transformers for text embeddings
  - Parallel processing for large datasets

- **High Performance**
  - Achieved **97% F1-score** for Information Theory classification
  - **95% accuracy** on Computational Linguistics
  - **93% balanced accuracy** for Computer Vision


## 🛠️ Technical Implementation

### Data Preprocessing
- Custom text cleaning pipeline to preprocess academic papers.
- This includes:
  - Removal of stopwords
  - Stemming
  - Handling special characters and formatting inconsistencies

### Sentence Embeddings
To convert the raw academic text into meaningful embeddings, we use **Sentence Transformers**. This method helps to capture semantic meaning more effectively than traditional vectorization techniques like TF-IDF.

1. **Initialize Sentence Transformer Model**
  
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2")


2. **Generate Embeddings**
   Using the model, we transform each academic paper's text into high-quality vector representations, which are used as input features for the classifier.


### Multi-Label Classification

The project evaluates different machine learning classifiers (KNN, Random Forest, LightGBM) for multi-label classification, where each academic paper can belong to multiple categories. These classifiers are compared based on their F1-score for each domain.

* **KNN (K-Nearest Neighbors)**: A simple yet effective method for multi-label classification.
* **Random Forest**: An ensemble method providing robust classification with better handling of imbalanced data.
* **LightGBM**: A gradient boosting model that achieved the highest performance across all domains.

