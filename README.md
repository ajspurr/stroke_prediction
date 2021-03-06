# Stroke Prediction
About 800,000 people in the United States have a stroke each year, making it a leading cause of serious long-term disability ([CDC](https://www.cdc.gov/stroke/facts.htm)). Fortunately, there are actions you can take to prevent or lower your chances of having a stroke such as eating healthy, maintaining a healthy weight, excercising, and abstaining from smoking. While everyone should follow these healthy habits, it can be helpful to identify individuals at high risk of stroke so they can pay especially close attention to their daily habits and make sure their chronic conditions (e.g. diabetes, hypertension) are well-controlled.

In this analysis, I explore the Kaggle [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset). I'll go through the major steps in Machine Learning to build and evaluate classification models to predict whether or not an individual is likely to have a stroke. This doesn't necessarily calculate a lifetime risk of stroke or chances of an acute stroke, but it can identify high-risk individuals who should take the preventive actions mentioned above.  

Origin of the data: the Kaggle user who posted the dataset, [fedesoriano](https://www.kaggle.com/fedesoriano), notes that this data comes from a confidential source and is to be used only for educational purposes.

## Analysis Highlights
- Exploratory Data Analysis: 
  - Dataset of 5110 individuals with features such as gender, age, BMI, and presence/absense of heart disease and hypertension
  - Highly imbalanced target: only 5% had a stroke
- Dealing with imbalanced binary classification
  - I compared Weighted Logistic Regression to Logistic Regression w/ SMOTE (Synthetic Minority Oversampling TEchnique)
- Choosing a Model
  - Compared Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost, KNN (all w/ SMOTE)
- Hyperparameter Tuning
  - Chose top three performers: Logistic Regression, SVM, and XGBoost
  - Used GridSearchCV to perform hyperparameter tuning optimized for recall and f1 score
- Evaluating Models
  - I chose to focus on recall as the primary metric of evaluation so as to not miss individuals at higher risk of stroke (I discuss the cost of poor precision at the end of the project). As such, the two best models were:
  - **Non-optimized Weighted Logistic Regression**
    - Recall of 98.2% 
      - Out of 4,088 individuals in the test set, missed 1 stroke out of 55
    - Precision of 7% 
      - 714 false positives in the same test set
  - **Optimized Weighted XGBoost**
    - Recall of 91.7%
      - Out of 4,088 individuals in the test set, missed 4.5 strokes out of 55 (the fraction is because this is an average of the cross-validated recalls)
    - Precision of 11.1%
      - 404 false positives in the same test set

## Programming Language and Resource Details
**Python Version:** 3.8.8

**Packages:** pandas, numpy, sklearn, imblearn, matplotlib, seaborn

**Resources:** Reference links embedded in appropriate sections

## EDA
Full code: [stroke_eda.py](/stroke_eda.py) <br> All figures: [stroke_prediction/output/eda](/output/eda)

1. Dataset summary below. Fortunately, there is minimal missing data. 

<p align="center"><img src="/output/eda/data_overview.png" width="600"/></p>
<p align="center"><img src="/output/eda/feature_summary_2.png" width="1200"/></p>

Most of the features are self-explanatory, and the categories of each categorical variable can be seen below. But I will clarify a few here:
- hypertension: presence or absence of hypertension
- heart_disease: presence or absence of heart disease
- avg_glucose_level: average blood glucose level
- stroke: '0' if patient did not experience a stroke, '1' if they did


<br>
2. Visualization of categorical variables below. Notably, the target variable (stroke) is highly imbalanced. 

<br><p align="center"><img src="/output/eda/combined_cat_counts.png" width="900"/></p>
<br>
3. Stroke prevalence noticeably higher in those with hypertension, heart disease, and who have ever been married. Noticeably lower in children and individuals who have never worked. To be clear, the percentage is within each subcategory. So, in this case, about 4.5% of females had a stroke.

<br><p align="center"><img src="/output/eda/combined_perc_stroke.png" width="900"/></p> 
<br>
4. The continuous variable 'age' is somewhat uniform but with a negative skew. Variable 'avg_glucose_level' has a strong positive skew and is somewhat bimodal. 'BMI' is fairly normal, with a slight positive skew. Features 'age' and 'avg_glucose_level' with observable difference in distribution in stroke vs. no stroke

<br><p align="center"><img src="/output/eda/combined_dist.png" width="900"/></p> 
<br>
5. Calculated correlation between continuous variables, between categorical variables (Cramer's V), and between continuous and categorical (Correlation Ratio). Variable pairs with correlation greater than 0.5 plotted below heatmaps. No correlations greater than 0.7. 

(Credit to Shaked Zychlinski for explaining categorical correlation in [his article](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9))

<br><p align="center"><img src="/output/eda/combined_corr.png" width="900"/></p> 

<p align="center"><img src="/output/eda/combined_corr_details.png" width="900"/></p> 

<br>

## Data Preparation
#### Missing Data
As noted above, 4% of BMI data missing. As these values are missing because they weren't recorded (as opposed to being values that don't exist) they can be imputed. I do not want to remove the rows completely and lose 4% of the data. I do not want to remove the entire column as BMI is likely helpful in predicting stroke. Will use sklearn SimpleImputer to replace missing data with mean BMI.

#### Scaling
Will scale (technically standardize) continuous variables using sklearn StandardScaler.

#### Categorical Variable Encoding
Ordinal encoding would not be useful as categorical variables are not ordinal. One-hot encoding is appropriate, especially since all features have low cardinality, so it won't need to create 10s or 100s of new columns. 

#### Data Cleaning
Fortunately, data was well-formatted. Replaced 0s and 1s in 'hypertension' and 'heart_disease' columns with more descriptive strings so that one-hot encoded columns are easier to interpret. 

#### Feature Engineering
Did not find obvious opportunities for feature engineering, but I can explore this in the future.

<br>

## Model Building
Full code: [stroke_modeling.py](/stroke_modeling.py) <br> All figures: [stroke_prediction/output/models](/output/models)
### Initial Model: Logistic Regression
Initially modelled with logistic regression without dealing with imbalanced dataset

<p align="center"><img src="/output/models/eval_metrics_log_reg.png" width="900"/></p> 
<img src="/output/models/metrics_log_reg.png" width="200"/>

This model has a 95% accuracy and AUROC of 85%. However, looking at the confusion matrix, you can see that it simply predicted that no one in the test dataset would have a stroke, and since only 5% had a stroke, it still resulted in 95% accuracy. This is why it is important to choose the correct metrics to measure model performance on imbalanced data. 

The recall (or sensitivity) was 0, which is unacceptable as it means this model would have missed 55 strokes in a set of about 1000 individuals. The precision (or positive predictive value) is actually undefined as it is calculated as TP/(TP + FP) which is 0/0 in this case. Recall and precision, as well as AUPRC and f1 by extension, focus more on the positive class (stroke) than the negative class, which makes them much more useful than accuracy and AUROC in this case. This is true for two reasons: 1. analytically, because this dataset is highly imbalanced towards the negative class, and 2. clinically, because you do not want to miss a stroke. 

### Dealing with an Imbalanced Dataset
#### Option 1: Optimize Model
Logistic regression works by fitting curves to the training dataset. It repeatedly changes the parameters of the curve to minimize the loss (error) of the model on the training dataset. Normally the errors for each target class (stroke vs. no stroke) are treated the same when it comes to using them to optimize the parameters. This clearly doesn't work well for imbalanced datasets. However, the sklearn LogisticRegression class has a hyperparatmer 'class_weight' which allows you to increase or decrease the weight of target classes. In this case, we will increase the weight of the 'stroke' class so that errors in prediction of stroke will lead to more updating of the model coefficients. And we will do the inverse for the 'no stroke' class. One way to choose the exact weights is to use the inverse of the class distribution in the training data. In this case, there are 194 individuals with stroke and 3894 without a stroke. So I will weight the 'no stroke' class as 194/3894 = 0.048, and weight the 'stroke' class as 3894/194 = 20.

(Credit to Jason Brownlee for explaining Weighted Logistic Regression in [this post](https://machinelearningmastery.com/cost-sensitive-logistic-regression/))

<p align="center"><img src="/output/models/eval_metrics_log_reg_weighted.png" width="900"/></p> 
<img src="/output/models/metrics_log_reg_w.png" width="280"/>

You can see a dramatic improvement in recall from 0 to 98%. This model missed only 1 stroke out of 55, which is great. The tradeoff is that there were 714 cases where the model wrongly predicited a stroke (compared to 0 before). This is reflected in the low precision of 7%. You can see this visually in the bottom right Precision/Recall vs. Threshold graph. The recall holds its high value for much longer, but the precision holds its low value longer as well. The weights can be tuned to optimize the balance between false positives and false negatives using sklearn GridSearchCV. This will be explored below. 

#### Option 2: Oversampling
Another way to deal with an imbalanced dataset is to either remove rows from the majority class (undersampling) or to add rows to the minority class (oversampling). This dataset isn't huge so removing data probably isn't the best choice. For oversampling, you can simply duplicate rows from the minority class, which doesn't add new information to the model, or you can synthesize new minority class data using SMOTE (Sythetic Minority Oversampling TEchnique). The technical details can be found in the source below, but the point is that it generates new minority class examples that are similar to the existing minority class data. 

(Credit to Jason Brownlee for explaining SMOTE in [this post](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/))

<p align="center"><img src="/output/models/pre_post_smote_pca.png" width="800"/></p> 

The plot to the left is a PCA of the original dataset. The plot to the right is a PCA after SMOTE was used. You can see there is now an equal number of positive and negative outcome cases. You can also see that the new samples aren't exact copies of one another, but are similar enough that they group together. Now I will use logistic regression to build a model, results below: 

<p align="center"><img src="/output/models/eval_metrics_log_reg_smote.png" width="900"/></p> 
<img src="/output/models/metrics_log_reg_smote.png" width="300"/>

Compared to weighted logistic regression, this model has a bit more balance of recall and precision. The recall decreased from 98% to 82% and the precision increased from 7% to 15%. It missed 10 strokes out of 55 (compared to missing just 1) but it only wrongly predicted a stroke 251 times (compared to 714). The f1 score (harmonic mean of recall and precision) increased from 0.13 to 0.26.

#### Putting It All Together
The heatmap below compares the performance metrics of logistic regression, weighted logistic regression, and logistic regression post-SMOTE. As mentioned above, the most important metric is recall, as you do not want to miss any stroke cases. In this regard, weighted logistic regression was by far the best. Logistic regression post-SMOTE had a significantly lower recall with a minimal increase in precision (and no increase in AUPRC, or average precision). This also resulted in a higher f1 value, which is the harmonic mean of recall and precision. 

<p align="center"><img src="/output/models/lr_all_metrics.png" width="900"/></p> 

## More Models
Explored more models post-SMOTE. Used cross-validation to calculate recall and f1 scores. 

<p align="center"><img src="/output/models/metrics_multiple_models_smote.png" width="900"/></p> 

As seen above, logistic regression and SVM had the highest recall, followed by gradient boosting, XGBoost, and KNN. Decision tree and random forest performed poorly both with recall and f1 score. Will move forward by performing hyperparameter tuning on the logistic regression, SVM, and XGBoost models:

## Hyperparameter Tuning - f1 score
I initially chose to optimize hyperparameters based on f1 score as this is a well-rounded measure of performance that incorporates recall and precision. The argument could be made to optimize solely on recall so as to minimize false negatives. This is explored at the end. 

### XGBoost
**Hyperparameter tuning values** (References: [1](https://www.mikulskibartosz.name/xgboost-hyperparameter-tuning-in-python-using-grid-search/), [2](https://towardsdatascience.com/binary-classification-xgboost-hyperparameter-tuning-scenarios-by-non-exhaustive-grid-search-and-c261f4ce098d), [3](https://machinelearningmastery.com/xgboost-for-imbalanced-classification/))
- max_depth: [2, 3, 4, 5, 6, 7, 8, 9]
- n_estimators: [60, 100, 140, 180]
- learning_rate: [0.1, 0.01, 0.05]
- scale_pos_weight: weights
  - Like the weighted logistic regression, I based the 'weights' on the inverse class distribution in the training data. In this case, there are 194 individuals with stroke and 3894 without a stroke. So the inverse class distribution is 3894/194 = 20. Then for hyperparameter tuning, I added 4 more values to try: the inverse class distribution +/- 25% and +/- 50%. 

#### New Optimized Models
- Weighted XGBoost (no SMOTE)
- XGBoost non-weighted with SMOTE
- Weighted XGBoost with SMOTE

#### Combined XGBoost Results
The first column is the orginal non-optimized XGBoost with SMOTE. Weighted XGBoost performed better than XGBoost with SMOTE in terms of both recall and f1 score. I wanted to test whether using both weights and SMOTE to deal an imbalanced dataset would improve results, but it did not perform better than just using weights. 

<p align="center"><img src="/output/models/combined_metrics_xgb.png" width="900"/></p> 

### Logistic Regression
**Hyperparameter tuning values** (References: [1](https://machinelearningknowledge.ai/hyperparameter-tuning-with-sklearn-gridsearchcv-and-randomizedsearchcv/), [2](https://machinelearningmastery.com/xgboost-for-imbalanced-classification/))
- C: np.logspace(-3, 3, 20)
- penalty: ['l2']
- class_weight: weights (weights derived similarly to XGBoost above)

#### New Optimized Models
- Weighted Logistic Regression (no SMOTE)
- Logistic Regression non-weighted with SMOTE

#### Combined Logistic Regression Results
The first column is the orginal non-optimized Weighted Logistic Regression. The second column is the orginal Logistic Regression with SMOTE. Weighted Logistic Regression had a better recall than Logistic Regression with SMOTE regardless of whether the model was optimized or not. This makes sense as the weighted models penalize false negatives much more than the non-weighted models. Optimization did not seem to improve recall or f1 either with Weighted Logistic Regression or Logistic Regression with SMOTE.

<p align="center"><img src="/output/models/combined_metrics_lr.png" width="900"/></p> 

### SVM
**Hyperparameter tuning values** (Reference: [1](https://machinelearningknowledge.ai/hyperparameter-tuning-with-sklearn-gridsearchcv-and-randomizedsearchcv/))
- C: [0.1, 1, 10, 100, 1000]
- gamma: [1, 0.1, 0.01, 0.001, 0.0001]
- kernel: ['rbf']
- class_weight: weights (weights derived similarly to XGBoost above)

#### New Optimized Models
- Weighted SVM (no SMOTE)
- SVM non-weighted with SMOTE

#### Combined SVM Results
The first column is the orginal non-optimized SVM with SMOTE. Unlike LR and XGB, Weighted SVM' recall was only marginally better than SVM SMOTE. 

<p align="center"><img src="/output/models/combined_metrics_svm.png" width="900"/></p> 
<br>

### Best Models Optimized for f1 score
The non-optimized Weighted LR still has the highest recall by far, although it has the worst f1. 
<p align="center"><img src="/output/models/combined_metrics_best_f1.png" width="900"/></p> 
<br>

## Hyperparameter Tuning - Recall
### Models Optimized for Recall
Repeated the same hyperparameter parameter tuning as above, but optized for recall instead of f1 score. As seen below, a couple of the weighted models reached a recall of 100%, catching all strokes. However, it looks like they may have predicted all individuals to have a stroke, as their accuracy was 5% and specificity was 0%. More precise hyperparameter tuning needs to be done. This can be explored in the future (see below)
<p align="center"><img src="/output/models/combined_metrics_recall.png" width="900"/></p> <br>

## Evaluating Models
- Choosing an appropriate metric was difficult. I would like to choose a model with the highest recall so as to not miss any high-risk individuals. However, high recall came at the cost of very low precision and therefore many false positives. For the sake of argument, let's say we are identifying high-risk individuals so that they can focus on healthy habits, as opposed to identifying them for some invasive testing. This limits the cost of false-positives to unecessary stress on individuals wrongly classified as high-risk. For this analysis, I will still prioritize recall, although this trade-off should be discussed with clinicians using the model. 
- Given the above assumptions, the two best models were:
  - **Non-optimized Weighted Logistic Regression**
    - Recall of 98.2% 
      - Out of 4,088 individuals in the test set, missed 1 stroke out of 55
    - Precision of 7% 
      - 714 false positives in the same test set
  - **Optimized Weighted XGBoost**
    - Recall of 91.7%
      - Out of 4,088 individuals in the test set, missed 4 strokes out of 55
    - Precision of 11.1%
      - 408 false positives in the same test set

## Potential Next Steps
- Improve predictions
  - Work through assumptions of logistic regression
  - Improve hyperparameter tuning of current models
    - Explore how to further optimize models for unbalanced data
    - More precise hyperparameter tuning
  - Explore feature engineering and feature importance
  - Outlier identification
  - Optimize and compare other models (e.g. KNN, Random Forest)
- Productionize chosen model
