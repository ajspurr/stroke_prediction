# Stroke-Prediction
Exploratory data analysis, data preprocessing, model training, parameter tuning, stroke prediction, and model evaluation using Kaggle stroke dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## EDA
Highlights from Exploratory Data Analysis ([code](/stroke_EDA.py), [all figures](/output))

1. Target variable, stroke, is highly imbalanced (bottom right figure)

<p align="center"><img src="/output/combined_cat_counts.png" width="900" align="middle"/></p>

2. Stroke prevalence noticeably higher in those with hypertension, heart disease, and who have ever been married. Noticeably lower in children

<p align="center"><img src="/output/combined_perc_stroke.png" width="900" align="middle"/></p> 

3. All continuous variables have somewhat normal or uniform distribution, 'avg_glucose_level' with a positive skew and somewhat bimodal and BMI with a slight positive skew. Variables 'age' and 'avg_glucose_level' with observable difference in distribution in stroke vs. no stroke

<p align="center"><img src="/output/combined_dist.png" width="900" align="middle"/></p> 
