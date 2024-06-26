#!/usr/bin/python

import sys
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.abspath(("../tools/")))
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### feature evaluation: deferred_income,loan_advances, expenses, from_messages, to_messages, from_poi_to_this_person, from_this_person_to_poi and director_fees have too many missing values or no relevant correlation
# Steps to Further Refine Feature Selection:
# Correlation Analysis: Check the correlation of each feature with the target variable to see which features have the strongest relationships.
# Feature Importance from Models: Train a preliminary model and check feature importances to see which features contribute the most to the model's performance.
# Missing Values Handling: Decide on a strategy for handling missing values, such as imputation, which can allow you to keep features that otherwise have missing data.
### preliminary feature list looks like this
features_list = [
    'poi',                          # Target variable
    'salary',                       # FEATURE: medium correlation to poi in heatmap analysis
    'bonus',                        # FEATURE: medium correlation to poi in heatmap analysis
    'total_payments',               # FEATURE: medium correlation to poi in heatmap analysis
    'total_stock_value',            # FEATURE: STRONG correlation to poi in heatmap analysis
    'exercised_stock_options',      # FEATURE: STRONG correlation to poi in heatmap analysis
    'long_term_incentive',          # FEATURE: medium correlation to poi in heatmap analysis
    'shared_receipt_with_poi',      # FEATURE: medium correlation to poi in heatmap analysis
    'restricted_stock',             # FEATURE: medium correlation to poi in heatmap analysis
    'from_this_person_to_poi',
    'from_messages',
] 

### Load the dictionary containing the dataset
with open("./final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Remove Outliers
data_dict.pop('TOTAL') # not relevant
data_dict.pop('BAXTER JOHN C') # very high total_stock_value
data_dict.pop('FREVERT MARK A') # very high total_payments
data_dict.pop('LAVORATO JOHN J') # very high bonus
data_dict.pop('BHATNAGAR SANJAY') # very high bonus
data_dict.pop('MARTIN AMANDA K') # very high total_payments
data_dict.pop('PAI LOU L') # very high total_stock_value
data_dict.pop('WHITE JR THOMAS E') # very high total_stock_value



# get rid of their dict/list structure and use dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')

# Identify columns that should not be converted to float
exclude_columns = ['email_address', 'poi']

# Convert all other columns to float
for col in df.columns:
    if col not in exclude_columns:
        df[col] = df[col].astype(float)

# print(df.describe().T)
# display(df.info())

# Handling missing values and trying which gives the best accuray initially
df_cleaned = df[features_list]
df_cleaned = df_cleaned.fillna(0) # best result
# df_cleaned = df_cleaned.fillna(df_cleaned.mean())  
# df_cleaned = df_cleaned.fillna(df_cleaned.median())  

# Feature Creation: Create new features // only worsens f1-score
df_cleaned['pct_to_poi'] = np.where(df_cleaned['from_messages'] != 0, df_cleaned['from_this_person_to_poi'] / df_cleaned['from_messages'], 0) # idea: pois get higher bonuses compared to salary
# df_cleaned['total_stock_value_vs_salary'] = np.where(df_cleaned['salary'] != 0, 
                                                    #  df_cleaned['bonus'] / df_cleaned['salary'], 0) # idea: pois get higher bonuses compared to salary
# df_cleaned['total_stock_value_vs_exercised'] = np.where(df_cleaned['salary'] != 0, 
#                                                         df_cleaned['total_payments'] / df_cleaned['salary'], 0) # with insider information pois might have exercised faster

features_list.remove('from_this_person_to_poi')
features_list.remove('from_messages')

# Add new features to the features list
features_list += ['pct_to_poi'] # 'total_stock_value_vs_exercised'] # 'exercised_stock_options_plus_total_stock_value' ] #, 'log_bonus', 'bonus_to_salary', 'total_payments_to_salary', 'stock_options_to_total_stock']

# Compute the correlation matrix
# correlation_matrix = df_cleaned[features_list].corr()

# ### Display the correlation matrix
# print(correlation_matrix)

# ### Find strong correlations between poi and other features 
# plt.figure(figsize=(12, 10))
# sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Matrix')
# plt.show()

# Convert the DataFrame back to a dictionary for featureFormat
data_dict = df_cleaned.to_dict(orient='index')

# Handling missing values with featureFormat and targetFeatureSplit from course
data = featureFormat(data_dict, features_list) 
y, X = targetFeatureSplit(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create a custom transformer for outlier detection
class OutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.clf = IsolationForest(contamination=self.contamination)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self

    def transform(self, X):
        outlier_labels = self.clf.predict(X)
        return np.column_stack((X, outlier_labels == -1))

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('outlier_detector', OutlierDetector(contamination=0.1)), ### makes result worse
    ('selector', SelectPercentile(f_classif)),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])


# Set up the parameter grid to search
param_grid = [
    {
        'selector__percentile': [20, 30, 40],
        'pca__n_components': [2, 3, 4],
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100],
        'classifier__max_features': ['sqrt'],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__bootstrap': [False],
    },
    # {
    #     'selector__percentile': [ 40, 50],
    #     'pca__n_components': [2, 3, 4],
    #     'classifier': [GradientBoostingClassifier(random_state=42)],
    #     'classifier__n_estimators': [100, 200, 300],
    #     'classifier__learning_rate': [0.01, 0.1, 0.2],
    #     'classifier__max_depth': [3, 4, 5],
    #     'classifier__min_samples_split': [2, 4],
    #     'classifier__min_samples_leaf': [1, 2],
    #     'classifier__subsample': [0.8, 0.9, 1.0],
    #     'classifier__max_features': ['sqrt', 'log2', None]
    # }
]

# Using cross-validation to get a more robust estimate of model performance
cv = StratifiedKFold(n_splits=30, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# Use the best estimator
best_clf = grid_search.best_estimator_

### Best parameters found: {'classifier': RandomForestClassifier(bootstrap=False, n_estimators=50, random_state=42), 'classifier__bootstrap': False, 
# 'classifier__max_depth': None, 'classifier__max_features': 'sqrt', 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 
# 'classifier__n_estimators': 50, 'pca__n_components': 4, 'selector__percentile': 50}

# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = best_clf.predict(X_test) # best_clf for using pipeline and param_grid
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ### Task 2: Remove outliers ### More than 'TOTAL'? 
# ### Task 3: Create new feature(s)
# ### Store to my_dataset for easy export below.
my_dataset = data_dict

# ### Task 4: Try a varity of classifiers
# ### Please name your classifier clf for easy export below.

# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info: 
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(best_clf, my_dataset, features_list)

# sorted_df = df_cleaned[features_list].sort_values(by="pct_to_poi", ascending=False)
# print(sorted_df[0:49])
