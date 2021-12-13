# Stroke Prediction
Exploratory data analysis, data preprocessing, model training, parameter tuning, stroke prediction, and model evaluation using Kaggle stroke dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## EDA
Highlights from Exploratory Data Analysis <br>Full code: [stroke_eda.py](/stroke_eda.py) <br> All figures: [stroke_prediction/output/eda](/output/eda)

1. Basic data exploration. Dataset has 5110 rows and 11 columns, a mix of numerical and categorical data types, and minimal missing data. 

<p align="center"><img src="/output/eda/feature_summary.png" width="600"/></p>
<br>
2. Visualization of categorical variables below. Notably, the target variable (stroke) is highly imbalanced. 

<p align="center"><img src="/output/eda/combined_cat_counts.png" width="900"/></p>
<br>
3. Stroke prevalence noticeably higher in those with hypertension, heart disease, and who have ever been married. Noticeably lower in children and individuals who have never worked.

<p align="center"><img src="/output/eda/combined_perc_stroke.png" width="900"/></p> 
<br>
4. All continuous variables have somewhat normal or uniform distribution, 'avg_glucose_level' with a positive skew and somewhat bimodal and BMI with a slight positive skew. Features 'age' and 'avg_glucose_level' with observable difference in distribution in stroke vs. no stroke

<p align="center"><img src="/output/eda/combined_dist.png" width="900"/></p> 
<br>
5. Calculated correlation between continuous variables, between categorical variables (Cramer's V), and between continuous and categorical (Correlation Ratio). Variable pairs with correlation greater than 0.5 plotted below heatmaps. No correlations greater than 0.7. 

<p align="center"><img src="/output/eda/combined_corr.png" width="900"/></p> 

<p align="center"><img src="/output/eda/combined_corr_details.png" width="900"/></p> 

<br>

## Data Preparation
#### Missing Data
As noted above, 4% of BMI data missing. As these values are missing because they weren't recorded (as opposed to being values that don't exist) they can be imputed. I do not want to remove the rows completely and lose 4% of the data. I do not want to remove the entire column as BMI is likely helpful in predicting stroke. Will use sklearn SimpleImputer to replace missing data with mean BMI.

#### Scaling
Will scale continuous variables using sklearn StandardScaler.

#### Categorical Variable Encoding
Ordinal encoding not useful as categorical are not ordinal. One-hot encoding is appropriate, especially since all features have low cardinality, so it won't need to create 10s or 100s of new columns. 

#### Data Cleaning
Fortunately, data was well-formatted. Replaced 0s and 1s in 'hypertension' and 'heart_disease' columns with more descriptive strings so that one-hot encoded columns are easier to interpret. 

#### Feature Engineering
Did not find obvious opportunities for feature engineering, but can explore this in the future.

<br>

## Model Building
Full code: [stroke_modeling.py](/stroke_modeling.py) <br> All figures: [stroke_prediction/output/models](/output/models)
### Initial Model: Logistic Regression
Initially modelled with logistic regression without dealing with imbalanced dataset

<p align="center"><img src="/output/models/eval_metrics_Log Reg.png" width="900"/></p> 
<img src="/output/models/metrics_log_reg.png" width="200"/>

This model has a 95% accuracy and AUROC of 85%. However, looking at the confusion matrix, you can see that it simply predicted that no one in the test dataset would have a stroke, and since only 5% had a stroke, it still resulted in 95% accuracy. This is why it is important to choose the correct metrics to measure model performance on imbalanced data. 

The recall (or sensitivity) was 0, which is unacceptable as it means this model would have missed 55 strokes in a set of about 1000 individuals. The precision (or positive predictive value) is actually undefined as it is calculated as TP/(TP + FP) which is 0/0 in this case. Recall and precision, as well as AUPRC and f1 by extension, focus more on the positive class (stroke) than the negative class, which makes them much more useful than accuracy and AUROC in this case. This is true for two reasons: 1. analytically, because this dataset is highly imbalanced towards the negative class, and 2. clinically, because you do not want to miss a stroke. 

### Dealing with an Imbalanced Dataset
#### Option 1: Optimize Model
Logistic regression works by fitting curves to the training dataset. It repeatedly changes the parameters of the curve to minimize the loss (error) of the model on the training dataset. Normally the errors for each target class (stroke vs. no stroke) are treated the same when it comes to using them to optimize the parameters. This clearly doesn't work well for imbalanced datasets. However, the sklearn LogisticRegression class has a hyperparatmer 'class_weight' which allows you to increase or decrease the weight of target classes. In this case, we will increase the weight of the 'stroke' class so that errors in prediction of stroke will lead to more updating of the model coefficients. And we will do the inverse for the 'not stroke' class. One way to choose the exact weights is to use the inverse of the class distribution in the training data. In this case, there are 194 individuals with stroke and 3894 without a stroke. So I will weight the 'no stroke' class as 194/3894 = 0.048, and weight the 'stroke' class as 3894/194 = 20.

(Credit to Jason Brownlee for explaining Weighted Logistic Regression in [this post](https://machinelearningmastery.com/cost-sensitive-logistic-regression/))

<p align="center"><img src="/output/models/eval_metrics_Log Reg (weighted).png" width="900"/></p> 
<img src="/output/models/metrics_log_reg_w.png" width="280"/>

You can see a dramatic improvement in recall from 0 to 98%. This model missed only 1 stroke out of 55, which is great. The tradeoff is that there were 714 cases where the model wrongly predicited a stroke. This is reflected in the low precision of 7%. You can see this visually in the bottom right Precision/Recall vs. Threshold graph. The recall holds its high value for much longer, but the precision holds its low value longer as well. The weights can be tuned to optimize the balance between false positives and false negatives using sklearn GridSearchCV. This can be explored in the future. 

#### Option 2: Oversampling
Another way to deal with an imbalanced dataset is to either remove rows from the majority class (undersampling) or to add rows to the minority class (oversampling). This dataset isn't huge so removing data probably isn't the best choice. For oversampling, you can simply duplicate rows from the minority class, which doesn't add new information to the model, or you can synthesize new minority class data using SMOTE (Sythetic Minority Oversampling TEchnique). The technical details can be found in the source below, but the point is that it generates new minority class examples that are similar to the existing minority class data. 

(Credit to Jason Brownlee for explaining SMOTE in [this post](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/))

<p align="center"><img src="/output/models/pre_post_smote_pca.png" width="800"/></p> 

The plot to the left is a PCA of the original dataset. The plot to the left is a PCA after SMOTE was used. You can see there is now an equal number of positive and negative outcome cases. You can also see that the new samples aren't exact copies of one another, but are close enough that they group together. Now I will use logistic regression to build a model, results below: 

<p align="center"><img src="/output/models/eval_metrics_Log Reg (SMOTE).png" width="900"/></p> 
<img src="/output/models/metrics_log_reg_smote.png" width="300"/>

Compared to weighted logistic regression, this model has a bit more balance of recall and precision. The recall decreased from 98% to 82% and the precision increased from 7% to 15%. It missed 10 strokes out of 55 (compared to missing just 1) but it only wrongly predicted a stroke 251 times (compared to 714).  

#### Putting It All Together
The heatmap below compares the performance metrics of logistic regression, weighted logistic regression, and logistic regression post-SMOTE. 

<p align="center"><img src="/output/models/lr_all_metrics.png" width="900"/></p> 
