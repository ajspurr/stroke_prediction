import pandas as pd
import numpy as np
from os import chdir
from pathlib import PureWindowsPath
import seaborn as sns 
import matplotlib.pyplot as plt

# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\Stroke-Prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/stroke-data.csv', index_col='id')

# ====================================================================================================================
# EXPLORATORY DATA ANALYSIS
# ====================================================================================================================
print()
print("DATASET SHAPE:")
print(dataset.shape)
print()
print("COLUMN INFO:")
print(dataset.info())
print()
pd.set_option("display.max_columns", len(dataset.columns))

print("BASIC INFORMATION NUMERICAL VARIABLES:")
print(dataset.describe())
print()
print("DATA SAMPLE:")
dataset.head()
print()
pd.reset_option("display.max_columns")
 
# =============================
# Explore target 
# =============================
# size includes NaN values, count does not
print("TARGET SUMMARY:")
print(dataset['stroke'].agg(['size', 'count', 'nunique', 'unique']))
print()
# Count of each unique value
print("VALUE COUNTS:")
print(dataset['stroke'].value_counts())
print()
# Total null values
print("TOTAL NULL VALUES:")
print(dataset['stroke'].isnull().sum())
print()

# =============================
# Explore features
# =============================
# Total missing values in each column 
col_missing_values = dataset.isnull().sum().to_frame()
col_missing_values = col_missing_values.rename(columns = {0:'missing_values'})

# Calculate percent missing in each column
col_missing_values['percent_missing'] = (col_missing_values['missing_values'] / len(dataset.index)) * 100
print("MISSING VALUES:")
print(col_missing_values)
print()

# Column 'bmi'  missing 201 values, about 4% of the total
# As 'bmi' values are missing because they weren't recorded (as opposed to being values that don't exist), 
# the missing values can be imputed. Or I could remove the rows completely. It wouldn't make sense to remove 
# the column as 'bmi' is likely a strong predictor
# Will address this later when building models

# Separate categorical and numerical features
categorical_cols = [cname for cname in dataset.columns if dataset[cname].dtype == "object"]
numerical_cols = [cname for cname in dataset.columns if not dataset[cname].dtype == "object"]

# See if there are any 'numerical' columns that actually contain encoded categorical data
num_uniques = dataset[numerical_cols].nunique()

# In this case there are 3 'numerical' columns with only 2 unique values, so they'll be moved to the categorical list
more_cat_cols = [cname for cname in num_uniques.index if  num_uniques[cname] < 3]
categorical_cols = categorical_cols + more_cat_cols
numerical_cols = [col for col in numerical_cols if col not in more_cat_cols]
#numerical_cols = list(set(numerical_cols) - set(more_cat_cols))

# Remove target variable 'stroke' from list of categorical variables, it can be analyzed on its own
categorical_cols.remove('stroke')

# =======================================================================================
# Visualize data
# =======================================================================================

# =============================
# Categorical variables
# =============================

# Categorical data bar charts
for col in categorical_cols:
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts()).set_title(col)
    plt.show()
sns.barplot(x=dataset['stroke'].value_counts().index, y=dataset['stroke'].value_counts()).set_title('stroke')
plt.show()
# Very few cases of stroke, hypertension, and heart disease. It will be interesting to see if they are correlated.

# Visualize relationship between categorical data and outcome
# This version has the categorical variable on the x-axis, each one split by how many have had strokes or not
for col in categorical_cols:
    sns.catplot(x=col, data=dataset, kind="count", hue="stroke")
    plt.show()
# Not very helpful visualization as the number of records with strokes is very low

# This version has stroke on the x-axis, each one split by how many records are in each categorical variable
for col in categorical_cols:
    sns.catplot(x="stroke", data=dataset, kind="count", hue=col)
    plt.show()
# A little better, but again, since number of strokes is so low, percentages may be better:


# Parameters: data = dataset, target = string representing column name of target variable, 
#             categorical_var = string representing column name of target variable
# Example: Target variable is binary variable 'stroke', categorical variable is 'smoking_status' which has 4 possible values
# Within 'stroke' subgroup, will display the percentage of records of each smoking status. Same to subgroup 'non-stroke'
def boxplot_percentage_of_target_category(data, target, categorical_var):
    # Create multi-index dataframe with primary index as the target, secondary index as categorical variable
    df_grouped = data.groupby([target, categorical_var])[categorical_var].count().to_frame()
    df_grouped = df_grouped.rename(columns = {categorical_var:'count'})

    # This code ensures that if a certain subcategory isn't present in the 'stroke' or 'no stroke' subset, 
    # it will be added, with a count of '0'
    df_grouped = df_grouped.unstack().fillna(0).stack().astype(int)

    # Add column which represents the categorical variable count as a percentage of that category of target 
    # (stroke vs. not stroke). I used range to give them unique values for debugging purposes
    df_grouped['percent_of_target_cat'] = range(len(df_grouped))

    # Loop through multi-index dataframe, giving the new column it's appropriate value
    for target_value in df_grouped.index.levels[0]:
        for categorical_var_value in df_grouped.index.levels[1]:
            df_grouped.loc[(target_value, categorical_var_value), 'percent_of_target_cat'] = df_grouped.loc[(target_value, categorical_var_value), 'count'] / df_grouped.loc[(target_value, slice(None)), :]['count'].sum()

    # Convert from multi-index to two columns with those index values 
    # This will add columns for target and categorical variable value, as it makes it easier to create a boxplot
    df_grouped = df_grouped.reset_index()
    
    # Plots figure with target as x-axis labels, each with a bar for each categorical variable
    plt.figure() # Ensures seaborn won't plot multiple figures on top of one another
    sns.barplot(x=df_grouped[target], y=df_grouped['percent_of_target_cat'], hue=df_grouped[categorical_var])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=categorical_var)
    
    # Plots figure with categorical variable as x-axis labels, each with a bar for each target
    plt.figure() # Ensures seaborn won't plot multiple figures on top of one another
    sns.barplot(x=df_grouped[categorical_var], y=df_grouped['percent_of_target_cat'], hue=df_grouped[target])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=target)
    
for cat_cols in categorical_cols:
    boxplot_percentage_of_target_category(dataset, 'stroke', cat_cols)
    plt.show()


# ==========================================================
# Continuous variables
# ==========================================================

# Numerical data histograms
for col in numerical_cols:
    #sns.distplot used to plot the histogram and fit line, but it's been deprecated to displot or histplot which don't 
    sns.distplot(dataset[col])
    plt.show()
    
    #Code below displays histogram with total counts (can use density=True for density instead) without fit line
    # plt.hist(dataset[col], density=False)
    # plt.title(col)
    # plt.show()

# Most have somewhat normal or uniform distribution, 'avg_glucose_level' with an obvious positive skew and somewhat 
# bimodal and BMI with a slight positive skew

# Distribution plots separated by stroke
for col in numerical_cols:
    sns.kdeplot(data=dataset[dataset.stroke==1], x=col, shade=True, alpha=1, label='stroke')
    sns.kdeplot(data=dataset[dataset.stroke==0], x=col, shade=True, alpha=0.5, label='no stroke')
    plt.legend()
    plt.show()


# =============================
# Correlation between continuous variables
# =============================

# Find correlation between variables
# np.trui sets all the values above a certain diagonal to 0, so we don't have redundant boxes
matrix = np.triu(dataset[numerical_cols].corr()) 
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, mask=matrix, cmap="rocket")
plt.show()

# You can also make a correlation matrix that includes the diagonal so that the color spectrum better 
# represents the more extreme values
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, cmap="rocket")
plt.show()

# Age has the highest correlation with other continuous variables

# Since age has the highest correlation with other variables, will plot those relationships
# First, scatterplots and lineplots showing relationship between age and other continuous variables
sns.scatterplot(data=dataset, x='age', y='bmi')
plt.show()
sns.lineplot(data=dataset, x='age', y='bmi')
plt.show()

sns.scatterplot(data=dataset, x='age', y='avg_glucose_level')
plt.show()
sns.lineplot(data=dataset, x='age', y='avg_glucose_level')
plt.show()

# Second, boxplots showing relationship between age and categorical variables
for cat_col in categorical_cols:
    sns.boxplot(data=dataset, x=cat_col, y='age')
    plt.show()

sns.boxplot(data=dataset, x='stroke', y='age')
plt.show()

# =============================
# Correlation between categorical variables?
# =============================

# =============================
# Risk of stroke by age
# =============================
stroke_rates = []

dataset['age'].min()
# Found that min age is 0.08. For the sake of the loop counter needing be an int, will say min_age = 1 
# as it calculates the risk of stroke at any age below the current age (so it won't ignore the age=0.08)
min_age = 1

dataset['age'].max()
# Found that max age in the dataset is 82.0, will call it 82 (an int)
max_age = 82

# Looping through each age to calculate risk of having stroke by the time someone reaches that age
for i in range(min_age, max_age):
    # Will spell it out for ease of understanding in the future
    # Current age calculating risk for
    age = i
    
    # In this dataset, number of strokes in anyone current age or younger
    num_strokes = dataset[dataset['age'] <= i]['stroke'].sum()
    
    # Total number of people in this dataset current age or younger
    num_people = len(dataset[dataset['age'] <= i])
    
    # Add the stroke rate to the list
    stroke_rates.append(num_strokes / num_people)

# Create line plot, the x-axis is technically the index of the value, but this is actually the age given the way the loop works
sns.lineplot(data=stroke_rates)
plt.xlabel("Age")
plt.ylabel("Cumulative Stroke Risk")
plt.show()

























