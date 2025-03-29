# NLP
 NLP Model Comparision

# 🧠 Text Classification: Traditional ML vs BERT

This project compares the performance of traditional machine learning models and a transformer-based BERT model on a binary text classification task using the **20 Newsgroups** dataset (specifically the *rec.sport.hockey* and *sci.space* categories).

---

## 🔍 Problem Statement

The goal is to classify news articles into one of two categories based on their textual content.

---

## 🛠️ Tools & Libraries

- Python, NumPy, pandas  
- **scikit-learn**: TF-IDF, Naive Bayes, Logistic Regression, SVM, Random Forest  
- **HuggingFace Transformers**: BERT, Trainer API  
- **PyTorch**: Custom Dataset & DataLoader  
- **Matplotlib & seaborn**: For visualizations

---

## 📊 Models Compared

| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Naïve Bayes        | ...      | ...       | ...    | ...      |
| Logistic Regression| ...      | ...       | ...    | ...      |
| SVM (Linear)       | ...      | **🔝**     | **🔝**  | **🔝**    |
| Random Forest      | ...      | ...       | ...    | ...      |
| **BERT (Transformer)** | **🔝**  | N/A       | N/A    | N/A      |

> 🔝 **Best Accuracy:** BERT  
> 🥇 **Best Precision, Recall, F1-score:** SVM

---

## ✅ Key Steps

1. **Data Loading & Cleaning**
   - Loaded 20 Newsgroups dataset from `sklearn.datasets`.
   - Filtered two categories: `rec.sport.hockey` and `sci.space`.
   - Removed headers, footers, and quotes for cleaner text.

2. **TF-IDF Vectorization**
   - Converted text data into numerical format using `TfidfVectorizer` with a 5,000-word vocabulary.

3. **Model Training**
   - Trained and evaluated four classical models:
     - Naïve Bayes
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
   - Metrics used: Accuracy, Precision, Recall, F1-score

4. **BERT Fine-tuning**
   - Used `bert-base-uncased` from HuggingFace.
   - Tokenized and encoded text data using a custom PyTorch `Dataset` class.
   - Fine-tuned for 0.5 epochs using HuggingFace's `Trainer`.

5. **Result Compilation**
   - Compiled all model scores into a pandas DataFrame for comparison.

---

## 🚀 Future Improvements

- Extend to multi-class classification
- Integrate more transformer models (e.g., RoBERTa, DistilBERT)
- Perform hyperparameter tuning & cross-validation
- Add error analysis & misclassification visualization

---

## 📌 Summary

This project demonstrates how transformer-based models like **BERT** outperform classical models in terms of accuracy for text classification. However, **SVM** achieved the highest precision, recall, and F1-score among traditional models — proving to be a strong, lightweight alternative.

---

## 📷 Sample Output

Feel free to include:
- Confusion matrix plots  
- Bar charts comparing model metrics  
- Training logs or screenshots  

---

## 🤝 Contributions

Pull requests, suggestions, and issues are always welcome!

---