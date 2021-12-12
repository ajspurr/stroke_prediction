# Stroke Prediction
Exploratory data analysis, data preprocessing, model training, parameter tuning, stroke prediction, and model evaluation using Kaggle stroke dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## EDA
Highlights from Exploratory Data Analysis <br>Full code: [stroke_eda.py](/stroke_eda.py) <br> All figures: [stroke_prediction/output/eda](/output/eda)

1. Basic data exploration. Dataset has 5110 rows and 11 columns, a mix of numerical and categorical data types, and minimal missing data. 

<p align="center"><img src="/output/eda/feature_summary.png" width="500"/></p>
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
Fortunately, data was well-formatted. Replaced 0s and 1s in 'hypertension' and 'heart_disease' columns with more descriptive string so that one-hot encoded columns are easier to interpret. 

#### Feature Engineering
Did not find obvious opportunities for feature engineering, but can explore this in the future.

<br>

## Model Building
### Dealing with Imbalanced Dataset
