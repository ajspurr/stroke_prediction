import pandas as pd
import numpy as np
from os import chdir
from pathlib import PureWindowsPath, Path
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import f1_score


# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\stroke_prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/stroke-data.csv', index_col='id')
output_dir = Path(project_dir, Path('./output/models'))

# ====================================================================================================================
# Organize features, split into training and validation datasets
# ====================================================================================================================
# Separate categorical and numerical variables
categorical_cols = [cname for cname in dataset.columns if dataset[cname].dtype == "object"]
numerical_cols = [cname for cname in dataset.columns if not dataset[cname].dtype == "object"]

# See if there are any 'numerical' columns that actually contain encoded categorical data
num_uniques = dataset[numerical_cols].nunique()

# In this case there are 3 'numerical' columns with only 2 unique values, so they'll be moved to the categorical list
more_cat_cols = [cname for cname in num_uniques.index if  num_uniques[cname] < 3]
categorical_cols = categorical_cols + more_cat_cols
numerical_cols = [col for col in numerical_cols if col not in more_cat_cols]

# Remove target variable 'stroke' from list of categorical variables
categorical_cols.remove('stroke')

# Will copy original dataframe to new dataframe for preprocessing
new_df = dataset.copy()

# Two 'numeric' columns are actually categorical, will map them to categorical values for one-hot encoding
hypertension_dict = {0:'normotensive', 1:'hypertensive'}
heart_disease_dict = {0:'no_heart_disease', 1:'heart_disease'}
new_df['hypertension'] = new_df['hypertension'].map(hypertension_dict)
new_df['heart_disease'] = new_df['heart_disease'].map(heart_disease_dict)

# ====================================================================================================================
# Data preprocessing function
# ====================================================================================================================

def create_pipeline(model_name, model, use_SMOTE):
    # Preprocessing for numerical data (SimpleImputer default strategy='mean')
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()), 
        ('scale', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
    ])
    
    if (use_SMOTE):
        # Bundle preprocessor then SMOTE then model
        oversample = SMOTE()
        my_pipeline = Pipeline_imb([
            ('preprocessor', preprocessor),
            ('oversample', oversample),
            (model_name, model)
        ]) 
    else:
        # Bundle preprocessor then model
        my_pipeline = Pipeline_imb([
            ('preprocessor', preprocessor),
            (model_name, model)
        ])
    return my_pipeline

# ====================================================================================================================
# Model evaluation function
# ====================================================================================================================
def evaluate_model(X_train, X_valid, y_train, y_valid, y_pred, pipeline_or_model, model_name, create_graphs=True):  
    # Accuracy
    accuracy = accuracy_score(y_valid, y_pred)
     
    # Confusion matrix heatmap
    # Includes counts and percetage of true outcome (so can compare performance in positive and negative cases)
    # Color based on percentage
    conmat = confusion_matrix(y_valid, y_pred)
    conmat_df = pd.DataFrame(conmat)
    # Create new confusion matrix converting the count to a percentage of true outcome
    conmat_df_perc = conmat_df.div(conmat_df.sum(axis=1), axis=0)
    # Labels for each box
    labels = ['True Neg','False Pos','False Neg','True Pos']
    counts = ["{0:0.0f}".format(value) for value in conmat.flatten()]
    percentages = ["{:.2%}".format(value) for value in conmat_df_perc.to_numpy().flatten()]
    label = (np.array([f'{v1}\n{v2}\n({v3})' for v1,v2,v3 in zip(labels,counts,percentages)])).reshape(2,2)
    # Rename columns and indeces as they become the heatmap axis labels
    conmat_df_perc.columns = ['No stroke', 'Stroke']
    conmat_df_perc.index  = ['No stroke', 'Stroke']
    
    if (create_graphs):
        #Create heatmap
        sns.heatmap(conmat_df_perc, annot=label, cmap="Blues", fmt="", vmin=0)
        plt.ylabel('True outcome')
        plt.xlabel('Predicted outcome')
        plt.title(f'Confusion Matrix ({model_name})')
        plt.show() 
    
    # ROC, AUC
    y_probs = pipeline_or_model.predict_proba(X_valid)
    y_probs = y_probs[:, 1]
    fpr, tpr, roc_thresholds = roc_curve(y_valid, y_probs)
    
    AUC = roc_auc_score(y_valid, y_probs)
    
    # Plot ROC
    if (create_graphs):
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.fill_between(fpr, tpr, facecolor='orange', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({model_name})')
        plt.text(0.95, 0.05, f'AUC = {AUC:.2f}', ha='right', fontsize=12, weight='bold', color='blue')
        plt.show()
    
    # Precision, recall, PRC, AUPRC
    precision, recall, prc_thresholds = precision_recall_curve(y_valid, y_probs)
    average_precision = average_precision_score(y_valid, y_probs)
    AUPRC = auc(recall, precision)
    
    # Calculate baseline for PRC plot (number of positive events over the total number of events)
    baseline = len(y_valid[y_valid==1]) / len(y_valid)
    
    if (create_graphs):
        # Plot PRC
        plt.plot(recall, precision, marker='.', label=f'AUPRC: {AUPRC:.2f}', color="blue")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision Recall Curve ({model_name})')
        plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline', color="orange")
        plt.legend()
        plt.show()
    
    if (create_graphs):
        # Plot precision and recall for each threshold
        plt.plot(prc_thresholds, precision[:-1], label='Precision',c='orange')
        plt.plot(prc_thresholds, recall[:-1],label='Recall',c='b')
        plt.title(f'Precision/Recall vs. Threshold ({model_name})')
        plt.ylabel('Precision/Recall Value')
        plt.xlabel('Thresholds')
        plt.legend()
        plt.ylim([0,1])
        plt.show()
    
    # F1 score
    f1 = f1_score(y_valid, y_pred)
    
    # Calculate other metrics commonly used in biomedical research
    TN, FP, FN, TP = list(map(float, counts))
    sensitivity = TP / (TP+FN) # Same as recall
    specificity = TN / (TN+FP)
    try:
        PPV = TP / (TP+FP) # Same as precision
    except ZeroDivisionError:
        PPV = 0
        print("While evaluating model " + model_name + ", encountered 'ZeroDivisionError' while calculating PPV, so setting PPV to zero")
    NPV = TN / (TN+FN)
    f1_manual = (2*TP) / ((2*TP) + FP + FN)
    
    # Figure out what threshold it being used for the above metrics
    # Combine corresponding thresholds, precisions, recalls into one dataframe
    # Technically precision and recall dataframes have one more value than thresholds df,
    # so the last value is just not included in the combination
    combined_df = pd.concat([pd.DataFrame(prc_thresholds), pd.DataFrame(precision), pd.DataFrame(recall)], axis=1, join='inner')
    combined_df.columns = ['threshold', 'precision', 'recall']
    # Selecting rows where the precision is very close to what I calculated above, then I can access the corresponding thresholds
    target_rows = combined_df.loc[(combined_df['precision'] > (PPV-0.00001)) & (combined_df['precision'] < (PPV+0.00001))]
    possible_thresholds = list(target_rows['threshold'])
    
    # Model performance metrics to be returned by this function
    metrics = {}
    metrics['Accuracy'] = np.round(accuracy, 4)
    metrics['Sensitivity (recall)'] = np.round(sensitivity, 4)
    metrics['Specificity'] = np.round(specificity, 4)
    metrics['PPV (precision)'] = np.round(PPV, 4)
    metrics['NPV'] = np.round(NPV, 4)
    metrics['AUROC'] = np.round(AUC, 4)
    metrics['Average precision'] = np.round(average_precision, 4)
    metrics['AUPRC'] = np.round(AUPRC, 4)
    metrics['F1'] = np.round(f1, 4)
    metrics['F1 (manual)'] = np.round(f1_manual, 4)
    metrics['Possible thresholds used'] = possible_thresholds
    
    
    return metrics, conmat_df

# ====================================================================================================================
# Evaluate Logistic Regression in detail
# ====================================================================================================================

# Separate target from predictors
y = new_df['stroke']
X = new_df.drop(['stroke'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing of training data and fit model
my_pipeline = create_pipeline('Log Reg', LogisticRegression(random_state=15))
fit = my_pipeline.fit(X_train, y_train)

# Get predictions
y_pred = my_pipeline.predict(X_valid)

results, conmat = evaluate_model(X_train, X_valid, y_train, y_valid, y_pred, my_pipeline, 'Log Reg')

# ====================================================================================================================
# Evaluate multiple models using cross validation scores (f1)
# ====================================================================================================================

# Creating a dictionary of models, keeping track of their pipelines and performance
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Gradient Boosting', 'XGBoost', 'KNN']

# Each dictionary key is the model name
models_dict = dict.fromkeys(model_names, None)

# The value of each item is another dictionary of model information
model_information_keys = ['Model', 'Pipeline', 'Predictions', 'CV Scores (f1)', 'CV Scores (recall)', 'Results']
for key in models_dict.keys():
    models_dict[key] = dict.fromkeys(model_information_keys, None)

# Create the actual model objects and add to dictionary
models_dict['Logistic Regression']['Model'] = LogisticRegression(random_state=15)
models_dict['Decision Tree']['Model'] = DecisionTreeClassifier(random_state=15)
models_dict['Random Forest']['Model'] = RandomForestClassifier(random_state=15)
models_dict['SVM']['Model'] = SVC(random_state=15, probability=True)
models_dict['Gradient Boosting']['Model'] = GradientBoostingClassifier(random_state=15)
models_dict['XGBoost']['Model'] = XGBClassifier(random_state=15, eval_metric='logloss', learning_rate = 0.054, use_label_encoder=False)
models_dict['KNN']['Model'] = KNeighborsClassifier()

# Create the pipeline for each model
for key in models_dict.keys():
    models_dict[key]['Pipeline'] = create_pipeline(key, models_dict[key]['Model'], use_SMOTE=True)

# Performing cross validation for each model
for key in models_dict.keys():
    models_dict[key]['CV Scores (f1)'] = cross_val_score(models_dict[key]['Pipeline'], X, y, cv=10, scoring='f1')

for key in models_dict.keys():
    models_dict[key]['CV Scores (recall)'] = cross_val_score(models_dict[key]['Pipeline'], X, y, cv=10, scoring='recall')

# Print mean CV scores for each model
print('\nMean f1 scores:')
for key in models_dict.keys():
    print(key, "{0:.4f}".format(models_dict[key]['CV Scores (f1)'].mean()))

print('\nMean recall scores:')
for key in models_dict.keys():
    print(key, "{0:.4f}".format(models_dict[key]['CV Scores (recall)'].mean()))

# To get full evaluation metrics on each model, need to fit first 
for key in models_dict.keys():
    models_dict[key]['Pipeline'].fit(X_train, y_train)
    models_dict[key]['Predictions'] = models_dict[key]['Pipeline'].predict(X_valid)

# Get full evaluation metrics on each model
for key in models_dict.keys():
    results, conmat = evaluate_model(X_train, X_valid, y_train, y_valid, models_dict[key]['Predictions'], models_dict[key]['Pipeline'], key, create_graphs=False)
    models_dict[key]['Results'] = results

# Combine most important results into one dataframe
final_metrics = ['Accuracy', 'Recall (CV)', 'Specificity', 'Precision (avg)', 'NPV', 'AUROC', 'f1 (CV)']
final_results = pd.DataFrame(columns=final_metrics, index=model_names)

for row in final_results.index:
    model_name = row
    model_data = models_dict[model_name]
    
    final_results_row = final_results.loc[row]
     
    final_results_row['Accuracy'] = model_data['Results']['Accuracy']
    final_results_row['Recall (CV)'] = np.round(model_data['CV Scores (recall)'].mean(), 4)
    final_results_row['Specificity'] = model_data['Results']['Specificity']
    final_results_row['Precision (avg)'] = model_data['Results']['Average precision']
    final_results_row['NPV'] = model_data['Results']['NPV']
    final_results_row['AUROC'] = model_data['Results']['AUROC']
    final_results_row['f1 (CV)'] = np.round(model_data['CV Scores (f1)'].mean(), 4)
  
# Display final results table
# pd.set_option("display.max_columns", len(final_results.columns))
# final_results
# pd.reset_option("display.max_columns")

# Create heatmap of final results to visualize best model, need to convert dataframe to numeric, for some reason it wasn't
final_results = final_results.apply(pd.to_numeric)
sns.heatmap(data=final_results, annot=True, cmap="Blues", fmt=".3")
plt.show()

# Logistic Regression Ranks close to the top in all metrics, let's look at the rest and reset the heatmap colors
new_results = final_results.copy().drop('Logistic Regression')
sns.heatmap(data=new_results, annot=True, cmap="Blues", fmt=".3")
plt.show()

# Random Forest and Decision Tree have the lowest Recall by a decent amount and lowest f1 by a bit, followed
# by KNN, will remove all three
# Recall is an important metric as you don't want to miss strokes
# f1 is an important metric in imbalanced datasets such as this one
new_results_2 = new_results.copy().drop(['Random Forest', 'Decision Tree', 'KNN'])
sns.heatmap(data=new_results_2, annot=True, cmap="Blues", fmt=".3")
plt.show()

# SVM with better recall but worse f1 than both boosting algorithms, will keep SVM and XGBoost
next_step_results = final_results.copy().drop(['Random Forest', 'Decision Tree', 'KNN', 'Gradient Boosting'])
sns.heatmap(data=next_step_results, annot=True, cmap="Blues", fmt=".3")
plt.show()

# ====================================================================================================================
# Hyperparameter tuning for Logistic Regression, SVM, XGBoost
# ====================================================================================================================
# ==========================================================
# Hyperparameter tuning XGBoost
# ==========================================================
# https://www.mikulskibartosz.name/xgboost-hyperparameter-tuning-in-python-using-grid-search/
estimator = XGBClassifier(objective='binary:logistic', nthread=4, seed=42, use_label_encoder=False, eval_metric='logloss')
estimator_pipe = create_pipeline('XGBoost', estimator, use_SMOTE=True)
parameters = {'XGBoost__max_depth': range (2, 10, 1), 'XGBoost__n_estimators': range(60, 220, 40), 'XGBoost__learning_rate': [0.1, 0.01, 0.05]}
grid_search = GridSearchCV(estimator=estimator_pipe, param_grid=parameters, scoring = 'f1', n_jobs = 10, cv = 10, verbose=True)

#estimator_pipe.get_params().keys()

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# {'XGBoost__learning_rate': 0.01,
#  'XGBoost__max_depth': 9,
#  'XGBoost__n_estimators': 100}
new_XGB_pipeline = grid_search.best_estimator_

new_XGB_pipeline.fit(X_train, y_train)
y_pred_new_XGB = new_XGB_pipeline.predict(X_valid)
results, conmat = evaluate_model(X_train, X_valid, y_train, y_valid, y_pred_new_XGB, new_XGB_pipeline, 'XGB (new)', create_graphs=False)

new_XGB_cv_f1 = cross_val_score(new_XGB_pipeline, X, y, cv=10, scoring='f1')
new_XGB_cv_recall = cross_val_score(new_XGB_pipeline, X, y, cv=10, scoring='recall')

print("Mean f1 CV score:" + str(np.round(new_XGB_cv_f1.mean(), 4)))
print("Mean recall CV score:" + str(np.round(new_XGB_cv_recall.mean(), 4)))

# Most metrics about the same other than CV recall, which had a large improvement from ~0.39 to ~0.50


# ==========================================================
# Hyperparameter tuning Logistic Regression
# ==========================================================
# Logistic regression
estimator = LogisticRegression(random_state=15)
estimator_pipe = create_pipeline('Logistic Regression', estimator, use_SMOTE=True)
parameters = {'Logistic Regression__C': np.logspace(-3, 3, 20), 'Logistic Regression__penalty': ['l2']}
grid_search = GridSearchCV(estimator=estimator_pipe, param_grid=parameters, scoring = 'f1', n_jobs = 10, cv = 10, verbose=True)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# {'Logistic Regression__C': 0.1623776739188721, 'Logistic Regression__penalty': 'l2'}

new_LR_pipeline = grid_search.best_estimator_

new_LR_pipeline.fit(X_train, y_train)
y_pred_new_LR = new_LR_pipeline.predict(X_valid)
results, conmat = evaluate_model(X_train, X_valid, y_train, y_valid, y_pred_new_LR, new_LR_pipeline, 'LR (new)', create_graphs=False)

new_LR_cv_f1 = cross_val_score(new_LR_pipeline, X, y, cv=10, scoring='f1')
new_LR_cv_recall = cross_val_score(new_LR_pipeline, X, y, cv=10, scoring='recall')
print("Mean f1 CV score:" + str(np.round(new_LR_cv_f1.mean(), 4)))
print("Mean recall CV score:" + str(np.round(new_LR_cv_recall.mean(), 4)))

# Most metrics about the same, it already performed better than other models

# ==========================================================
# Hyperparameter tuning SVM
# ==========================================================


#grid = GridSearchCV(svm, param_grid, refit = True, verbose =0,cv=10)


# Logistic regression
estimator = SVC(random_state=15, probability=True)
estimator_pipe = create_pipeline('SVM', estimator, use_SMOTE=True)
parameters = {'SVM__C': [0.1, 1, 10, 100, 1000], 
              'SVM__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'SVM__kernel': ['rbf']} 
grid_search = GridSearchCV(estimator=estimator_pipe, param_grid=parameters, refit = True, scoring = 'f1', n_jobs = 10, cv = 10, verbose=True)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# {'SVM__C': 0.1, 'SVM__gamma': 0.1, 'SVM__kernel': 'rbf'}

new_SVM_pipeline = grid_search.best_estimator_

new_SVM_pipeline.fit(X_train, y_train)
y_pred_new_SVM = new_SVM_pipeline.predict(X_valid)
results, conmat = evaluate_model(X_train, X_valid, y_train, y_valid, y_pred_new_SVM, new_SVM_pipeline, 'SVM (new)', create_graphs=False)

new_SVM_cv_f1 = cross_val_score(new_SVM_pipeline, X, y, cv=10, scoring='f1')
new_SVM_cv_recall = cross_val_score(new_SVM_pipeline, X, y, cv=10, scoring='recall')
print("Mean f1 CV score:" + str(np.round(new_SVM_cv_f1.mean(), 4)))
print("Mean recall CV score:" + str(np.round(new_SVM_cv_recall.mean(), 4)))

# Recall (CV) highly improved from 0.47 to ~0.79, other metrics similar to before




