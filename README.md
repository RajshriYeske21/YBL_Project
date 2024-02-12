# YBL_Project
# Financial Market News Sentiment Analysis

## Objective:
The objective of this project is to use machine learning models for analyzing sentiments in financial market news.

## Data Source:
The dataset used for this analysis is available at [GitHub](https://github.com/YBIFoundation/Dataset/blob/main/Financial%20Market%20News.csv).

## Import Libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Import Data:
dataset = pd.read_csv("https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Financial%20Market%20News.csv", encoding="ISO-8859-1")

Describe Data:
dataset.head()
dataset.info()
dataset.describe()

Data Visualization:
dataset.shape
dataset.columns

Data Preprocessing and Feature Extraction:
news = [' '.join(str(x) for x in dataset.iloc[row, 2:27]) for row in range(len(dataset.index))]
X = news

Feature text conversion to Bag of Words:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(lowercase=True, ngram_range=(1,1))
X = cv.fit_transform(X)

Define Target Variable (y) and Feature Variables (X):
y = dataset['Label']

Train and Split:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=2529)

Modelling:
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

Prediction:
y_pred = rf.predict(X_test)

Evaluation Matrix:
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

Conclusion:
This README provides an overview of the project, including its objective, data source, methodology, and evaluation metrics. For detailed implementation and code, please refer to the respective sections.


3. **Save README.md File:**
    Save the changes you made in your text editor or IDE.

4. **Upload to GitHub:**
    Push your code repository to GitHub, including the `README.md` file.

5. **Verify on GitHub:**
    Visit your GitHub repository to ensure that the `README.md` file is visible and properly formatted.

This README will help users understand your project and its implementation details when they visit your GitHub repository.
