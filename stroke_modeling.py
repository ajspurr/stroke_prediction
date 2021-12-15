import pandas as pd
import numpy as np
from os import chdir
from pathlib import PureWindowsPath, Path
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Pipeline_imb

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc, average_precision_score

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
# Visualization helper functions
# ====================================================================================================================
# Create 2d array of given size, used for figures with gridspec
def create_2d_array(num_rows, num_cols):
    matrix = []
    for r in range(0, num_rows):
        matrix.append([0 for c in range(0, num_cols)])
    return matrix

# Initialize figure, grid spec, axes variables
def initialize_fig_gs_ax(num_rows, num_cols, figsize=(16, 8)):
    # Create figure, gridspec, and 2d array of axes/subplots with given number of rows and columns
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax_array = create_2d_array(num_rows, num_cols)
    gs = fig.add_gridspec(num_rows, num_cols)

    # Map each subplot/axis to gridspec location
    for r in range(len(ax_array)):
        for c in range(len(ax_array[r])):
            ax_array[r][c] = fig.add_subplot(gs[r,c])

    # Flatten 2d array of axis objects to iterate through easier
    ax_array_flat = np.array(ax_array).flatten()
    
    return fig, gs, ax_array_flat

# Standardize image saving parameters
def save_image(dir, filename, dpi=300, bbox_inches='tight'):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches)
# ====================================================================================================================
# Data preprocessing function via pipeline
# ====================================================================================================================
def create_pipeline(model_name, model, use_SMOTE):
    # Preprocessing for numerical data (SimpleImputer default strategy='mean')
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()), 
        ('scale', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Not relevant in this dataset, but keeping for future application
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
    ])
    
    if (use_SMOTE):
        # Bundle preprocessor, SMOTE, model
        oversample = SMOTE(random_state=15)
        my_pipeline = Pipeline_imb([
            ('preprocessor', preprocessor),
            ('oversample', oversample),
            (model_name, model)
        ]) 
    else:
        # Bundle preprocessor and model
        my_pipeline = Pipeline_imb([
            ('preprocessor', preprocessor),
            (model_name, model)
        ])
    return my_pipeline

# ====================================================================================================================
# Model evaluation function
# ====================================================================================================================
# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels
def evaluate_model(X_train, X_valid, y_train, y_valid, y_pred, pipeline_or_model, model_name, 
                   model_display_name, create_graphs=True, combine_graphs=True, round_results=3):  
    # =============================
    # Accuracy
    # =============================
    accuracy = accuracy_score(y_valid, y_pred)
     
    # =============================
    # Confusion matrix heatmap: includes counts and percetage of true outcome, so can
    # compare performance in positive and negative cases (color based on percentage)
    # =============================
    conmat = confusion_matrix(y_valid, y_pred)
    conmat_df = pd.DataFrame(conmat)
    # Create new confusion matrix converting the count to a percentage of true outcome
    conmat_df_perc = conmat_df.div(conmat_df.sum(axis=1), axis=0)
    
    # =============================
    # ROC, AUC
    # =============================
    # Rather than predict positive or negative outcome, calculate probability of outcome being positive
    y_probs = pipeline_or_model.predict_proba(X_valid)
    y_probs = y_probs[:, 1]
    
    # Calculate False Positive Rates and True Positive Rates for predictions with score >= thresholds[i]
    fpr, tpr, roc_thresholds = roc_curve(y_valid, y_probs)
    
    # Calculate AUC for ROC
    AUC = roc_auc_score(y_valid, y_probs)
    
    # =============================
    # Precision, recall, PRC, AUPRC
    # =============================
    # Calculate precision, recall for each threshold 
    precision, recall, prc_thresholds = precision_recall_curve(y_valid, y_probs)
    
    # Calculate average presicion and AUC or PRC, which should be the same
    average_precision = average_precision_score(y_valid, y_probs)
    AUPRC = auc(recall, precision)
    
    # Calculate baseline for PRC plot (number of positive events over the total number of events)
    baseline = len(y_valid[y_valid==1]) / len(y_valid)

    # =============================
    # F1 score
    # =============================
    f1 = f1_score(y_valid, y_pred)
    
    # =============================
    # Calculate other metrics commonly used in biomedical research
    # =============================
    #TN, FP, FN, TP = list(map(float, counts))
    TN, FP, FN, TP = conmat.flatten()
    sensitivity = TP / (TP+FN) # Same as recall
    specificity = TN / (TN+FP)
    try:
        PPV = TP / (TP+FP) # Same as precision
    except:
        PPV = 0
        print("While evaluating model " + model_display_name + ", encountered 'ZeroDivisionError' while calculating PPV, so setting PPV to zero")
    NPV = TN / (TN+FN)
    f1_manual = (2*TP) / ((2*TP) + FP + FN)
    
    # =============================
    # Determine out what threshold it being used for the above metrics (specifically precision)
    # =============================
    # Combine corresponding thresholds, precisions, recalls into one dataframe
    # Technically precision and recall dataframes have one more value than thresholds df,
    # so the last value is just not included in the combination
    combined_df = pd.concat([pd.DataFrame(prc_thresholds), pd.DataFrame(precision), pd.DataFrame(recall)], axis=1, join='inner')
    combined_df.columns = ['threshold', 'precision', 'recall']
    # Selecting rows where the precision is very close to what I calculated above, then I can access the corresponding thresholds
    target_rows = combined_df.loc[(combined_df['precision'] > (PPV-0.00001)) & (combined_df['precision'] < (PPV+0.00001))]
    possible_thresholds = list(target_rows['threshold'])
    
    # =============================
    # Model performance metrics to be returned by this function
    # =============================
    metrics = {}
    metrics['Accuracy'] = np.round(accuracy, round_results)
    metrics['Sensitivity (recall)'] = np.round(sensitivity, round_results)
    metrics['Specificity'] = np.round(specificity, round_results)
    metrics['PPV (precision)'] = np.round(PPV, round_results)
    metrics['NPV'] = np.round(NPV, round_results)
    metrics['AUROC'] = np.round(AUC, round_results)
    metrics['Average precision'] = np.round(average_precision, round_results)
    metrics['AUPRC'] = np.round(AUPRC, round_results)
    metrics['F1'] = np.round(f1, round_results)
    metrics['F1 (manual)'] = np.round(f1_manual, round_results)   
    metrics['Possible thresholds used'] = list(map(lambda x: round(x, round_results), possible_thresholds))
    
    # =============================
    # Plot metrics
    # =============================
    if (create_graphs):
        if (combine_graphs):
            plot_model_metrics_combined(model_name, model_display_name, conmat, conmat_df_perc, fpr, tpr, AUC, precision, recall, prc_thresholds, AUPRC, baseline)
        else:
            plot_model_metrics(model_name, model_display_name, conmat, conmat_df_perc, fpr, tpr, AUC, precision, recall, prc_thresholds, AUPRC, baseline)
    
    return metrics, conmat_df

# Takes evalution metrics from evaluate_model() and plots confusion matrix, ROC, PRC, and precision/recall vs. threshold
# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels
def plot_model_metrics(model_name, model_display_name, conmat, conmat_df_perc, fpr, tpr, AUC, precision, recall, prc_thresholds, AUPRC, baseline):
    # =============================
    # Heatmap of confusion matrix
    # =============================
    # Labels for each box
    labels = ['True Neg','False Pos','False Neg','True Pos']
    counts = ["{0:0.0f}".format(value) for value in conmat.flatten()]
    percentages = ["{:.2%}".format(value) for value in conmat_df_perc.to_numpy().flatten()]
    label = (np.array([f'{v1}\n{v2}\n({v3})' for v1,v2,v3 in zip(labels,counts,percentages)])).reshape(2,2)
    # Rename columns and indeces as they become the heatmap axis labels
    conmat_df_perc.columns = ['No stroke', 'Stroke']
    conmat_df_perc.index  = ['No stroke', 'Stroke']
    
    #Create heatmap (hide colorbar (cbar) as the percentages are already displayed)
    sns.heatmap(conmat_df_perc, annot=label, cmap="Blues", fmt="", vmin=0, cbar=False)
    plt.ylabel('True outcome')
    plt.xlabel('Predicted outcome')
    plt.title(f'{model_display_name} Confusion Matrix')
    plt.show() 
    
    # =============================
    # ROC, AUC
    # =============================    
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.fill_between(fpr, tpr, facecolor='orange', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_display_name} ROC Curve (AUROC = {AUC:.2f})')
    #plt.text(0.95, 0.05, f'AUC = {AUC:.2f}', ha='right', fontsize=12, weight='bold', color='blue')
    plt.show()
    
    # =============================
    # Precision, recall, PRC, AUPRC
    # =============================    
    # Plot PRC
    plt.plot(recall, precision, marker='.', label='model', color="blue") # label=f'AUPRC: {AUPRC:.2f}'
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_display_name} Precision Recall Curve (AUPRC: {AUPRC:.2f})')
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline', color="orange")
    plt.legend()
    plt.show()

    # Plot precision and recall for each threshold
    plt.plot(prc_thresholds, precision[:-1], label='Precision', c='orange')
    plt.plot(prc_thresholds, recall[:-1],label='Recall', c='b')
    plt.title(f'{model_display_name} Precision/Recall vs. Threshold')
    plt.ylabel('Precision/Recall Value')
    plt.xlabel('Thresholds')
    plt.legend()
    plt.ylim([0,1])
    plt.show()

# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels
def plot_model_metrics_combined(model_name, model_display_name, conmat, conmat_df_perc, fpr, tpr, AUC, precision, recall, prc_thresholds, AUPRC, baseline):
    # Create figure, gridspec, list of axes/subplots mapped to gridspec location
    fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=2, num_cols=2, figsize=(14, 8))
   
    # =============================
    # Heatmap of confusion matrix
    # =============================
    # Labels for each box
    labels = ['True Neg','False Pos','False Neg','True Pos']
    counts = ["{0:0.0f}".format(value) for value in conmat.flatten()]
    percentages = ["{:.2%}".format(value) for value in conmat_df_perc.to_numpy().flatten()]
    label = (np.array([f'{v1}\n{v2}\n({v3})' for v1,v2,v3 in zip(labels,counts,percentages)])).reshape(2,2)
    # Rename columns and indeces as they become the heatmap axis labels
    conmat_df_perc.columns = ['No stroke', 'Stroke']
    conmat_df_perc.index  = ['No stroke', 'Stroke']
    
    #Create heatmap (hide colorbar (cbar) as the percentages are already displayed)
    axis = ax_array_flat[0]
    sns.heatmap(conmat_df_perc, annot=label, cmap="Blues", fmt="", vmin=0, cbar=False, ax=axis)
    axis.set_ylabel('True outcome')
    axis.set_xlabel('Predicted outcome')
    axis.set_title('Confusion Matrix') 
    
    # =============================
    # ROC, AUC
    # =============================   
    axis = ax_array_flat[1]
    axis.plot(fpr, tpr, color='orange', label='ROC')#, ax=axis)
    axis.plot([0, 1], [0, 1], color='darkblue', linestyle='--')#, ax=axis)
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])
    axis.fill_between(fpr, tpr, facecolor='orange', alpha=0.7)
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_title(f'ROC Curve (AUROC = {AUC:.2f})')
    #plt.text(0.95, 0.05, f'AUC = {AUC:.2f}', ha='right', fontsize=12, weight='bold', color='blue')
    
    # =============================
    # Precision, recall, PRC, AUPRC
    # =============================    
    # Plot PRC
    axis = ax_array_flat[2]
    axis.plot(recall, precision, marker='.', label='model', color="blue")#, ax=axis)
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.set_title(f'Precision Recall Curve (AUPRC: {AUPRC:.2f})')
    axis.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline', color="orange")#, ax=axis)
    axis.legend()

    # Plot precision and recall for each threshold
    axis = ax_array_flat[3]
    axis.plot(prc_thresholds, precision[:-1], label='Precision', c='orange')#, ax=axis)
    axis.plot(prc_thresholds, recall[:-1],label='Recall', c='b')#, ax=axis)
    axis.set_title('Precision/Recall vs. Threshold')
    axis.set_ylabel('Precision/Recall Value')
    axis.set_xlabel('Thresholds')
    axis.legend()
    axis.set_ylim([0,1])
    
    # Finalize figure formatting and export
    fig.suptitle(f'{model_display_name} Evaluation Metrics', fontsize=24)
    #fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    plt.subplots_adjust(hspace=0.3, wspace=0.2) # Increase spacing between plots if tight_layout doesn't work
    save_filename = 'eval_metrics_' + model_name
    save_image(output_dir, save_filename, bbox_inches='tight')
    plt.show()
    
# ====================================================================================================================
# Data preprocessing function without using pipeline
# ====================================================================================================================
def manual_preprocess(X_train, X_valid):
    # =============================
    # Numerical preprocessing
    # =============================
    X_train_num = X_train[numerical_cols]
    X_valid_num = X_valid[numerical_cols]
    
    # Imputation (default strategy='mean')
    num_imputer = SimpleImputer()
    imputed_X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    imputed_X_valid_num = pd.DataFrame(num_imputer.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # Scaling
    ss = StandardScaler()
    imputed_scaled_X_train_num = pd.DataFrame(ss.fit_transform(imputed_X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    imputed_scaled_X_valid_num = pd.DataFrame(ss.transform(imputed_X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # =============================
    # Categorical preprocessing
    # =============================
    X_train_cat = X_train[categorical_cols]
    X_valid_cat = X_valid[categorical_cols]
    
    # Imputation
    cat_imputer = SimpleImputer(strategy='most_frequent')
    imputed_X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train_cat), columns=X_train_cat.columns, index=X_train_cat.index)
    imputed_X_valid_cat = pd.DataFrame(cat_imputer.transform(X_valid_cat), columns=X_valid_cat.columns, index=X_valid_cat.index)
    
    # One-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train_cat), index=X_train_cat.index, columns=OH_encoder.get_feature_names_out())
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_X_valid_cat), index=X_valid_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_train_processed = pd.concat([imputed_scaled_X_train_num, OH_cols_train], axis=1)
    X_valid_processed = pd.concat([imputed_scaled_X_valid_num, OH_cols_valid], axis=1)
    
    return X_train_processed, X_valid_processed

# ====================================================================================================================
# Initial modeling of imbalanced data with Logistic Regression
# Compare with two methods of rectifying imblanced data (weighted logistic regression and SMOTE)
# ====================================================================================================================
# Separate target from predictors
y = new_df['stroke']
X = new_df.drop(['stroke'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# =============================
# Model with logistic regression without accounting for target imbalance
# =============================
# Preprocess data
X_train_processed, X_valid_processed = manual_preprocess(X_train, X_valid)

# Fit logistic regression model
log_reg = LogisticRegression(random_state=15)
fit = log_reg.fit(X_train_processed, y_train)

# Make predictions
y_pred = log_reg.predict(X_valid_processed)

# Evaluate model
results, conf_mat = evaluate_model(X_train_processed, X_valid_processed, y_train, y_valid, y_pred, log_reg, 'log_reg', 'Log Reg')

# =============================
# Model with weighted logistic regression
# =============================
# Quantify target imbalance in training dataset
num_pos_target = sum(y_train == 1)
num_neg_target = sum(y_train == 0)
perc_pos_target = round(num_pos_target / (num_pos_target + num_neg_target), 4)
ratio_pos_to_neg = round(num_pos_target / num_neg_target, 4)
ratio_neg_to_pos = round(num_neg_target / num_pos_target, 4)

# Set up weights
weights = {0:ratio_pos_to_neg, 1:ratio_neg_to_pos}

# Fit weighted logistic regression model
log_reg_w = LogisticRegression(class_weight=weights, random_state=15)
fit_w = log_reg_w.fit(X_train_processed, y_train)

# Make predictions
y_pred_w = log_reg_w.predict(X_valid_processed)

# Evaluate model
results_w, conf_mat_w = evaluate_model(X_train_processed, X_valid_processed, y_train, y_valid, y_pred_w, log_reg_w, 'log_reg_weighted', 'Log Reg (weighted)')

# =============================
# Model after using SMOTE
# =============================
smt = SMOTE(random_state=15)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train_processed, y_train)

# Fit logistic regression to oversampled data
log_reg_s = LogisticRegression(random_state=15)
fit_s = log_reg_s.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_s = log_reg_s.predict(X_valid_processed)

# Evaluate model
results_s, conmat_s = evaluate_model(X_train_resampled, X_valid_processed, y_train_resampled, y_valid, y_pred_s, log_reg_s, 'log_reg_smote', 'Log Reg (w/ SMOTE)')


# =============================
# Dataset before and after SMOTE
# =============================   
# Pre-SMOTE with PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train_processed)
principal_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'], index=X_train_processed.index)
final_df = pd.concat([principal_df, y_train], axis = 1)

# Post-SMOTE with PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train_resampled)
principal_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'], index=X_train_resampled.index)
final_df_s = pd.concat([principal_df, y_train_resampled], axis = 1)


# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(10, 5))

# PCA scatterplots
# Pre-SMOTE
axis = ax_array_flat[0]
sns.scatterplot(x=final_df['PC1'], y=final_df['PC2'], hue=final_df['stroke'], s=30, ax=axis)
axis.set_title('PCA Pre-SMOTE')
current_handles, current_labels = axis.get_legend_handles_labels()
num_stroke = sum(y_train == 1)
num_no_stroke = sum(y_train == 0)
new_legend_labels = ['No stroke (count: ' + str(num_no_stroke) + ')', 'Stroke (count: ' + str(num_stroke) + ')']
axis.legend(handles=current_handles, labels=new_legend_labels, loc='upper right')

#Post-SMOTE
axis = ax_array_flat[1]
sns.scatterplot(x=final_df_s['PC1'], y=final_df_s['PC2'], hue=final_df_s['stroke'], s=30, ax=axis)
axis.set_title('PCA Post-SMOTE')
current_handles, current_labels = axis.get_legend_handles_labels()
num_stroke = sum(y_train_resampled == 1)
num_no_stroke = sum(y_train_resampled == 0)
new_legend_labels = ['No stroke (count: ' + str(num_no_stroke) + ')', 'Stroke (count: ' + str(num_stroke) + ')']
axis.legend(handles=current_handles, labels=new_legend_labels, loc='upper right')


# Finalize figure formatting and export
fig.suptitle('Pre- and Post-SMOTE PCA', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#plt.subplots_adjust(hspace=0.3, wspace=0.2) # Increase spacing between plots if tight_layout doesn't work
save_filename = 'pre_post_smote_pca'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# =============================
# Combine above 3 model performance metrics into one dataframe then heatmap
# =============================
# Create dictionary of model performance metrics
lr_model_names = ['LR', 'LR (weighted)', 'LR (SMOTE)']

# The dictionary keys are the model names
lr_models_dict = dict.fromkeys(lr_model_names, None)
lr_models_dict['LR'] = results
lr_models_dict['LR (weighted)'] = results_w
lr_models_dict['LR (SMOTE)'] = results_s

# Combine most important results into one dataframe
lr_metrics = ['Accuracy', 'Sensivity (recall)', 'Specificity', 'AUROC', 'PPV (precision)', 'NPV', 'AUPRC', 'f1']
lr_final_results = pd.DataFrame(columns=lr_metrics, index=lr_model_names)

for row in lr_final_results.index:
    df_row = lr_final_results.loc[row]
    df_row['Accuracy'] = lr_models_dict[row]['Accuracy']
    df_row['Sensivity (recall)'] = lr_models_dict[row]['Sensitivity (recall)']
    df_row['Specificity'] = lr_models_dict[row]['Specificity']
    df_row['AUROC'] = lr_models_dict[row]['AUROC']
    
    PPV = lr_models_dict[row]['PPV (precision)']
    if (np.isnan(PPV)):
        df_row['PPV (precision)'] = 0
    else:
        df_row['PPV (precision)'] = PPV
    df_row['NPV'] = lr_models_dict[row]['NPV']
    df_row['AUPRC'] = lr_models_dict[row]['AUPRC']
    df_row['f1'] = lr_models_dict[row]['F1']
lr_final_results = lr_final_results.apply(pd.to_numeric)

# Display in heatmap (transposed the df first)
sns.heatmap(data=lr_final_results.T, annot=True, cmap="Blues", fmt=".3")
plt.yticks(rotation=0)  # Rotate y-tick labels to be horizontal
plt.title('Comparison of Three Logistic Regression Models')
#save_filename = 'lr_all_metrics'
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# Other output
print("LOGISTIC REGRESSION METRICS\n")
print(lr_final_results.loc['LR'])
print("WEIGHTED LOGISTIC REGRESSION METRICS\n")
print(lr_final_results.loc['LR (weighted)'])
print("LOGISTIC REGRESSION w/SMOTE METRICS\n")
print(lr_final_results.loc['LR (SMOTE)'])
    

# ====================================================================================================================
# Evaluate multiple models using cross validation scores for f1 and recall
# ====================================================================================================================
# Separate target from predictors
y = new_df['stroke']
X = new_df.drop(['stroke'], axis=1)

# =============================
# Create a dictionary of models, keeping track of their pipelines and performance
# =============================
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Gradient Boosting', 'XGBoost', 'KNN']

# The dictionary keys are the model names
models_dict = dict.fromkeys(model_names, None)

# The value of each item is another dictionary of model information
model_information_keys = ['Model', 'Pipeline', 'Predictions', 'CV Scores (f1)', 'CV Scores (recall)', 'Results']
for key in models_dict.keys():
    models_dict[key] = dict.fromkeys(model_information_keys, None)

# =============================
# Create and evaluate models, storing everything in dictionary created above
# =============================
# Create the model objects and add to dictionary
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


# Perform cross validation (f1 and recall) for each model 
for key in models_dict.keys():
    scores = cross_validate(models_dict[key]['Pipeline'], X, y, cv=10, scoring=['f1', 'recall'])
    models_dict[key]['CV Scores (f1)'] = scores['test_f1']
    models_dict[key]['CV Scores (recall)'] = scores['test_recall']

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
    results, conmat = evaluate_model(X_train, X_valid, y_train, y_valid, models_dict[key]['Predictions'], models_dict[key]['Pipeline'], key, key, create_graphs=False)
    models_dict[key]['Results'] = results

# Debugging - see all thresholds used
for key in models_dict.keys():
    print(key)
    print(models_dict[key]['Results']['Possible thresholds used'])
    print()

# Combine most important results into one dataframe
final_metrics = ['Accuracy', 'Sensitivity (recall, CV)', 'Specificity', 'AUROC', 'PPV (precision)', 'NPV', 'AUPRC', 'f1 (CV)']
final_results = pd.DataFrame(columns=final_metrics, index=model_names)

for row in final_results.index:
    model_name = row
    model_data = models_dict[model_name]
    
    final_results_row = final_results.loc[row]
     
    final_results_row['Accuracy'] = model_data['Results']['Accuracy']
    final_results_row['Sensitivity (recall, CV)'] = round(model_data['CV Scores (recall)'].mean(), 4)
    final_results_row['Specificity'] = model_data['Results']['Specificity']
    final_results_row['AUROC'] = model_data['Results']['AUROC']
    final_results_row['PPV (precision)'] = model_data['Results']['PPV (precision)']
    final_results_row['NPV'] = model_data['Results']['NPV']
    final_results_row['AUPRC'] = model_data['Results']['AUPRC']
    final_results_row['f1 (CV)'] = round(model_data['CV Scores (f1)'].mean(), 4)
  
# Display final results table
# pd.set_option("display.max_columns", len(final_results.columns))
# final_results
# pd.reset_option("display.max_columns")

# Create heatmap of final results to visualize best model, need to convert dataframe to numeric, for some reason it wasn't
final_results = final_results.apply(pd.to_numeric)
sns.heatmap(data=final_results.T, annot=True, cmap="Blues", fmt=".3")
plt.xticks(rotation=30, horizontalalignment='right')  # Rotate y-tick labels to be horizontal# Rotate x-axis tick labels so they don't overlap
plt.title('Model Metrics post-SMOTE')
save_filename = 'metrics_multiple_models_smote'
save_image(output_dir, save_filename, bbox_inches='tight')
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
plt.yticks(rotation=0)  # Rotate y-tick labels to be horizontal
plt.show()

# SVM with better recall but worse f1 than both boosting algorithms, will keep SVM and XGBoost
next_step_results = final_results.copy().drop(['Random Forest', 'Decision Tree', 'KNN', 'Gradient Boosting'])
sns.heatmap(data=next_step_results, annot=True, cmap="Blues", fmt=".3")
plt.yticks(rotation=0)  # Rotate y-tick labels to be horizontal
plt.show()

# ====================================================================================================================
# Hyperparameter tuning for Logistic Regression, SVM, XGBoost
# ====================================================================================================================
# =============================
# GridSearchCV attributes
# =============================
# The optimized parameters found by GridSearchCV: grid_search_obj.best_params_
# The pipeline or model with optimized parameters: grid_search_obj.best_estimator_
# The entirety of the gridsearch results: grid_search_obj.cv_results_
# The best score of whatever metric you chose to select the best parameters (what you set 'refit' to): grid_search_obj.best_score_
# The model parameter names to be used when making the GridSearchCV object: model_obj.get_params().keys()

# =============================
# Function organizing GridSearchCV results
# =============================
# Function assumes scoring=['f1', 'recall'] and that refit='f1'
# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels
# The recall and precision that are returned are the mean cross-validated values
def gridsearch_results(model_name, model_display_name, estimator, param_grid, scoring, refit, n_jobs=10, cv=10, verbose=True):
    # Create GridSearch object and fit data
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, refit=refit, n_jobs=n_jobs, cv=cv, verbose=verbose)
    grid_search.fit(X_train, y_train)
    
    # Access scores of best estimator
    results = grid_search.cv_results_
    
    # Gets the index of the best f1 score (to be used later as well)
    best_estimator_f1_index = np.nonzero(results['rank_test_f1'] == 1)[0][0]
    
    # Uses the index to select the best f1 score, which was used to select the best estimator
    best_estimator_f1 = results['mean_test_f1'][best_estimator_f1_index]
    
    # To get the recall score of the best estimator, use the same index as the best f1, since that corresponds to the same estimator
    best_estimator_recall = results['mean_test_recall'][best_estimator_f1_index]
    
    # Using optimal model from GridSearch results, fit and run model again so that I can run my results functions
    pipeline_gs = grid_search.best_estimator_
    pipeline_gs.fit(X_train, y_train)
    y_pred_gs = pipeline_gs.predict(X_valid)
    
    # Get results using my function
    results_gs, conmat_gs = evaluate_model(X_train, X_valid, y_train, y_valid, y_pred_gs, pipeline_gs, model_name, model_display_name, create_graphs=False)
    
    # Combine most important results into one dataframe
    return_metrics = ['Accuracy', 'Sensitivity (recall, CV)', 'Specificity', 'AUROC', 'PPV (precision)', 'NPV', 'AUPRC', 'f1 (CV)']
    return_results = pd.DataFrame(columns=return_metrics, index=[model_display_name])
    return_results['Accuracy'] = results_gs['Accuracy']    
    return_results['Sensitivity (recall, CV)'] = best_estimator_recall
    return_results['Specificity'] = results_gs['Specificity']
    return_results['AUROC'] = results_gs['AUROC']
    return_results['PPV (precision)'] = results_gs['PPV (precision)']
    return_results['NPV'] = results_gs['NPV']
    return_results['AUPRC'] = results_gs['AUPRC']
    return_results['f1 (CV)'] = best_estimator_f1
    
    return grid_search, return_results

# =======================================================================================
# Hyperparameter tuning XGBoost
# =======================================================================================
# https://www.mikulskibartosz.name/xgboost-hyperparameter-tuning-in-python-using-grid-search/
# https://towardsdatascience.com/binary-classification-xgboost-hyperparameter-tuning-scenarios-by-non-exhaustive-grid-search-and-c261f4ce098d

# =============================
# Weighted XGBoost with hyperparameter tuning without SMOTE
# =============================
model_name = 'xgboost_w'
model_display_name = 'Weighted XGBoost'

xgb_model = XGBClassifier(objective='binary:logistic', nthread=4, seed=15, use_label_encoder=False, eval_metric='logloss')
xgb_pipeline = create_pipeline(model_name, xgb_model, use_SMOTE=False)

# One way to optimize XGBoost for an imbalanced dataset is to set the 'scale_pos_weight' to the inverse of the class distribution
# https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
num_pos_target = sum(y_train == 1) # minority class
num_neg_target = sum(y_train == 0) # majority class
inv_class_dist = num_neg_target / num_pos_target

# Will use gridsearch to test inv_class_dist plus nearby values
weights = [inv_class_dist*0.5, inv_class_dist*0.75, inv_class_dist, inv_class_dist*1.25, inv_class_dist*1.5]

xgb_parameters = {model_name + '__max_depth': range (2, 10, 1), 
                  model_name + '__n_estimators': range(60, 220, 40), 
                  model_name + '__learning_rate': [0.1, 0.01, 0.05],
                  model_name + '__scale_pos_weight': weights}

grid_search_obj_xgb, return_results_xgb = gridsearch_results(model_name=model_name, 
                                                             model_display_name=model_display_name, 
                                                             estimator=xgb_pipeline, 
                                                             param_grid=xgb_parameters, 
                                                             scoring=['f1', 'recall'], refit='f1', 
                                                             n_jobs=10, cv=10, verbose=True)

# =============================
# XGBoost with hyperparameter tuning with SMOTE
# =============================
model_name = 'xgboost_t_s'
model_display_name = 'XGBoost (SMOTE)'

xgb_model_smote = XGBClassifier(objective='binary:logistic', nthread=4, seed=15, use_label_encoder=False, eval_metric='logloss')
xgb_pipeline_smote = create_pipeline(model_name, xgb_model_smote, use_SMOTE=True)

xgb_smote_parameters = {model_name + '__max_depth': range (2, 10, 1), 
                        model_name + '__n_estimators': range(60, 220, 40), 
                        model_name + '__learning_rate': [0.1, 0.01, 0.05]}

grid_search_obj_xgb_s, return_results_xgb_s = gridsearch_results(model_name=model_name, 
                                                                 model_display_name=model_display_name,
                                                                 estimator=xgb_pipeline_smote, 
                                                                 param_grid=xgb_smote_parameters, 
                                                                 scoring=['f1', 'recall'], refit='f1', 
                                                                 n_jobs=10, cv=10, verbose=True)


# =============================
# Weighted XGBoost with hyperparameter tuning WITH SMOTE
# =============================
model_name = 'xgboost_w_s'
model_display_name = 'Weighted XGBoost (SMOTE)'

xgb_model_w_s = XGBClassifier(objective='binary:logistic', nthread=4, seed=15, use_label_encoder=False, eval_metric='logloss')
xgb_pipeline_w_s  = create_pipeline(model_name, xgb_model_w_s, use_SMOTE=True)

# One way to optimize XGBoost for an imbalanced dataset is to set the 'scale_pos_weight' to the inverse of the class distribution
# https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
num_pos_target = sum(y_train == 1) # minority class
num_neg_target = sum(y_train == 0) # majority class
inv_class_dist = num_neg_target / num_pos_target

# Will use gridsearch to test inv_class_dist plus nearby values
weights = [inv_class_dist*0.5, inv_class_dist*0.75, inv_class_dist, inv_class_dist*1.25, inv_class_dist*1.5]

xgb_w_s_parameters = {model_name + '__max_depth': range (2, 10, 1),
                      model_name + '__n_estimators': range(60, 220, 40), 
                      model_name + '__learning_rate': [0.1, 0.01, 0.05], 
                      model_name + '__scale_pos_weight': weights}

grid_search_obj_xgb_w_s_, return_results_xgb_w_s_ = gridsearch_results(model_name=model_name, 
                                                             model_display_name=model_display_name, 
                                                             estimator=xgb_pipeline_w_s, 
                                                             param_grid=xgb_w_s_parameters, 
                                                             scoring=['f1', 'recall'], refit='f1', 
                                                             n_jobs=10, cv=10, verbose=True)


# =============================
# Combine all XGBoost results
# =============================
combined_xgb = pd.concat([final_results.T['XGBoost'], return_results_xgb.T, return_results_xgb_s.T, return_results_xgb_w_s_.T], axis=1, join='inner')
combined_xgb.columns = ['Non-optimized XGB SMOTE', 'Optimized Weighted XGB', 'Optimized XGB SMOTE', 'Optimized Weight XGB SMOTE']
sns.heatmap(data=combined_xgb, annot=True, cmap="Blues", fmt=".3")
plt.xticks(rotation=30, horizontalalignment='right')  # Rotate y-tick labels to be horizontal# Rotate x-axis tick labels so they don't overlap
plt.title('XGBoost Combined Metrics (optimized for f1 score)')
save_filename = 'combined_metrics_xgb'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()


# =======================================================================================
# Hyperparameter tuning Logistic Regression
# =======================================================================================
# https://machinelearningknowledge.ai/hyperparameter-tuning-with-sklearn-gridsearchcv-and-randomizedsearchcv/
# =============================
# Weighted Logistic Regression with hyperparameter tuning without SMOTE
# =============================
model_name = 'log_reg_w'
model_display_name = 'Weighted Log Reg'

lr_model = LogisticRegression(random_state=15)
lr_pipeline = create_pipeline(model_name, lr_model, use_SMOTE=False)

# Calculate baseline weight parameters as in Weighted Logistic Regression much higher in code
num_pos_target = sum(y_train == 1) # minority class
num_neg_target = sum(y_train == 0) # majority class
ratio_pos_to_neg = num_pos_target / num_neg_target

# Set up weights
weights = [{0:ratio_pos_to_neg, 1:(1/ratio_pos_to_neg)}]
new_ratio = ratio_pos_to_neg*0.5
weights.append({0:new_ratio, 1:(1/new_ratio)})
new_ratio = ratio_pos_to_neg*0.75
weights.append({0:new_ratio, 1:(1/new_ratio)})
new_ratio = ratio_pos_to_neg*1.25
weights.append({0:new_ratio, 1:(1/new_ratio)})
new_ratio = ratio_pos_to_neg*1.5
weights.append({0:new_ratio, 1:(1/new_ratio)})


lr_parameters = {model_name + '__C': np.logspace(-3, 3, 20), 
                 model_name + '__penalty': ['l2'],
                 model_name + '__class_weight' : weights}

grid_search_obj_lr, return_results_lr = gridsearch_results(model_name=model_name, 
                                                             model_display_name=model_display_name, 
                                                             estimator=lr_pipeline, 
                                                             param_grid=lr_parameters, 
                                                             scoring=['f1', 'recall'], refit='f1', 
                                                             n_jobs=10, cv=10, verbose=True)

return_results_lr.T


# =============================
# Logistic Regression with hyperparameter tuning with SMOTE
# =============================
model_name = 'log_reg_t_s'
model_display_name = 'Log Reg (SMOTE)'

lr_t_s_model = LogisticRegression(random_state=15)
lr_t_s_pipeline = create_pipeline(model_name, lr_t_s_model, use_SMOTE=True)

lr_t_s_parameters = {model_name + '__C': np.logspace(-3, 3, 20), 
                 model_name + '__penalty': ['l2']}

grid_search_obj_lr_t_s, return_results_lr_t_s = gridsearch_results(model_name=model_name, 
                                                             model_display_name=model_display_name, 
                                                             estimator=lr_t_s_pipeline, 
                                                             param_grid=lr_t_s_parameters, 
                                                             scoring=['f1', 'recall'], refit='f1', 
                                                             n_jobs=10, cv=10, verbose=True)

return_results_lr_t_s.T

# =============================
# Combine all Log Reg results
# =============================
combined_lr = pd.concat([final_results.T['Logistic Regression'], return_results_lr.T, return_results_lr_t_s.T], axis=1, join='inner')
combined_lr.columns = ['Non-optimized LR SMOTE', 'Optimized Weighted LR', 'Optimized LR SMOTE']
sns.heatmap(data=combined_lr, annot=True, cmap="Blues", fmt=".3")
plt.xticks(rotation=30, horizontalalignment='right')  # Rotate y-tick labels to be horizontal# Rotate x-axis tick labels so they don't overlap
plt.title('LR Combined Metrics (optimized for f1 score)')
#save_filename = 'combined_metrics_lr'
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()



# =======================================================================================
# Hyperparameter tuning SVM
# =======================================================================================

#grid = GridSearchCV(svm, param_grid, refit = True, verbose =0,cv=10)

# SVM
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

# Recall (cross val) highly improved from 0.47 to ~0.79, other metrics similar to before




