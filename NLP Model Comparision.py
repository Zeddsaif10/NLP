import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset (Using a sample dataset from sklearn for text classification)
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset='all', categories=['rec.sport.hockey', 'sci.space'], remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize Models
models = {
    "Naïve Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train and Evaluate Models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

results = {}
for name, model in models.items():
    acc, prec, rec, f1 = evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    results[name] = [acc, prec, rec, f1]

# BERT Implementation
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {**{k: v.squeeze() for k, v in encoding.items()}, 'labels': torch.tensor(self.labels[idx])}

bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)

train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)

training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=0.5, per_device_train_batch_size=5, per_device_eval_batch_size=5,
    evaluation_strategy="epoch", save_strategy="epoch", logging_dir='./logs', logging_steps=10,
)

trainer = Trainer(
    model=bert_model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
)

trainer.train()
bert_results = trainer.evaluate()

# Store BERT Results
results["BERT"] = [bert_results['eval_accuracy'], None, None, None]

# Convert results to DataFrame and Display
results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-score"]).T
import ace_tools_open as tools
tools.display_dataframe_to_user(name="Model Comparison Results", dataframe=results_df)