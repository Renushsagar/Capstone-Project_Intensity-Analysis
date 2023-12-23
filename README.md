Intensity Analysis (Build your own model using NLP and Python)
-
**Objective:-** Develop an intelligent system using NLP to predict the intensity in the text reviews. By analyzing various parameters and process data, the system will predict the intensity where its happiness, angriness or sadness. This predictive capability will enable to proactively optimize their processes, and improve overall customer satisfaction.

**Note:-** For detail understanding please check attched updated pdf file

**Problem Overview:-**
This project implements a machine learning pipeline to predict emotions or sentiments. It employs Logistic Regression, KNN, Random Forest, Support Vector Machine, and Naive Bayes classifiers, with TF-IDF vectorization. Performance is evaluated using classification report and accuracy.

**Pre-Requisitions:-**
pip install pandas,
pip install nltk,
nltk.download('punkt'),
nltk.download('stopwords'),
pip install imbalanced-learn,
pip install scikit-learn,
pip install xgboost

**Data Description:-**
Three datasets are utilized:
happiness.csv, sadness.csv, angriness.csv
combined shape of dataset is (2039, 2).

**Methodology**
**Preprocessing using NLP:-**
•	It’s a crucial step in natural language processing (NLP) to clean and transform raw text data into a format suitable for analysis

**Feature Engineering & Feature Selection:-**
•	Feature engineering in the context of TF-IDF vectorization involve creating new features or selecting a subset of existing features to improve model performance

**Dealing With Imbalanced Target Using Undersampling:-**
•	imbalanced means one class has significantly fewer instances than the other, so we choose undersampling technique to balance.

**Model Training:-**
•	Splitting the combined dataset into 80% training and 20% testing.
•	Employing TF-IDF for feature extraction.
•	Model selections: Logistic Regression, KNN, Random Forest, Support Vector Machine, and Naive Bayes

**Proposed Approach:-**
•	Data Exploration,
•	Handling Missing values and duplicates (if present),
•	Data Visualization,
•	Detailed preprocessing steps,
•	Model selection and implementation,
•	Dealing with imbalanced dataset using undersampling,
•	Hyperparameter tuning and Cross Validation,
•	Model Evaluation,
•	Test and Validate the final trained model.

**Conclusion:-**
Model with the highest test accuracy is SVM with 70.13%. However, Logistic Regression model has a relatively good test accuracy (69.70%) and a lower accuracy difference (22.29%) compared to other models. Therefore, considering both accuracy and the accuracy difference, the Logistic Regression model appears to be a reasonable choice.
