import pandas as pd
import numpy as np
from os import chdir
from pathlib import PureWindowsPath, Path
import seaborn as sns 
import matplotlib.pyplot as plt

# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\Stroke-Prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/stroke-data.csv', index_col='id')
output_dir = Path(project_dir, Path('./output'))



# ====================================================================================================================
# EXPLORATORY DATA ANALYSIS
# ====================================================================================================================
print("\nDATASET SHAPE:")
print(dataset.shape)
print("\nCOLUMN INFO:")
print(dataset.info())
pd.set_option("display.max_columns", len(dataset.columns))

print("\nBASIC INFORMATION NUMERICAL VARIABLES:")
print(dataset.describe())
print("\nDATA SAMPLE:")
print(dataset.head())
pd.reset_option("display.max_columns")
 
# =============================
# Explore target 
# =============================
# size includes NaN values, count does not
print("\nTARGET SUMMARY:")
print(dataset['stroke'].agg(['size', 'count', 'nunique', 'unique']))
# Count of each unique value
print("\nVALUE COUNTS:")
print(dataset['stroke'].value_counts())
# Total null values
print("\nTOTAL NULL VALUES:")
print(dataset['stroke'].isnull().sum())

# =============================
# Explore features
# =============================
# Total missing values in each column 
col_missing_values = dataset.isnull().sum().to_frame()
col_missing_values = col_missing_values.rename(columns = {0:'missing_values'})

# Calculate percent missing in each column
col_missing_values['percent_missing'] = (col_missing_values['missing_values'] / len(dataset.index)) * 100
print("\nMISSING VALUES:")
print(col_missing_values)

# Column 'bmi'  missing 201 values, about 4% of the total, will address later

# Separate categorical and numerical features
categorical_cols = [cname for cname in dataset.columns if dataset[cname].dtype == "object"]
numerical_cols = [cname for cname in dataset.columns if not dataset[cname].dtype == "object"]

# See if there are any 'numerical' columns that actually contain encoded categorical data
num_uniques = dataset[numerical_cols].nunique()

# In this case there are 3 'numerical' columns with only 2 unique values: hypertension, heart_disease, 
# and stroke (which is the target). They will be moved to the categorical list
more_cat_cols = [cname for cname in num_uniques.index if  num_uniques[cname] < 3]
categorical_cols = categorical_cols + more_cat_cols
numerical_cols = [col for col in numerical_cols if col not in more_cat_cols]

# Map 'hypertension' and 'heart_disease' numeric values to categorical values
hypertension_dict = {0:'normotensive', 1:'hypertensive'}
heart_disease_dict = {0:'no_heart_disease', 1:'heart_disease'}
dataset['hypertension'] = dataset['hypertension'].map(hypertension_dict)
dataset['heart_disease'] = dataset['heart_disease'].map(heart_disease_dict)

# Remove target variable 'stroke' from list of categorical variables, it can be analyzed on its own
categorical_cols.remove('stroke')

# =======================================================================================
# Visualize data
# =======================================================================================

def save_image(dir, filename, dpi=300, bbox_inches='tight'):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches)

# Format column names for figure labels
formatted_cols = {}
for col in dataset.columns:
    formatted_cols[col] = col.replace('_', ' ').title()
formatted_cols['bmi'] = 'BMI'

def format_col(col_name):
    return formatted_cols[col_name]

# ==========================================================
# Categorical variables
# ==========================================================

# Categorical data bar charts, total count of each category
for col in categorical_cols:
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts())
    plt.title(format_col(col))
    save_filename = 'counts_' + col
    save_image(output_dir, save_filename, bbox_inches='tight')
    plt.show()
sns.barplot(x=dataset['stroke'].value_counts().index, y=dataset['stroke'].value_counts()).set_title('Stroke')
save_image(output_dir, 'counts_stroke')
plt.show()
# Very few cases of stroke, hypertension, and heart disease. It will be interesting to see if they are correlated.

# =============================
# Relationship between categorical data and outcome
# =============================
# These bar charts use the categorical variable on the x-axis, each one split by stroke vs. no stroke
for col in categorical_cols:
    sns.catplot(x=col, data=dataset, kind="count", hue="stroke", legend=False)
    plt.title('Stroke Count by ' + format_col(col))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='stroke')
    save_filename = 'count_stroke_by_' + col
    save_image(output_dir, save_filename)
    plt.show()
# Not very helpful visualization as the number of records with stroke is very low

# This version has stroke on the x-axis, each one split by how many records are in each categorical variable
for col in categorical_cols:
    sns.catplot(x="stroke", data=dataset, kind="count", hue=col)
    plt.title(format_col(col) + ' Count by Stroke')
    plt.show()
# A little better, but again, since number of strokes is so low, percentages may be better:


# Function: boxplot_percentage()
# Parameters: data = dataframe representing the dataset, target = string representing column name of target variable, 
#             categorical_var = string representing column name of categorical variable
# Example: Target variable is binary variable 'stroke', categorical variable is 'smoking_status' which has 4 possible values
# This will create two boxplots. The first will have 'stroke' on the x-axis, subdivided by 'smoking_status' category.
# Within 'stroke' group, will display the percentage of records of each smoking status. The second graph reverses the 
# categorical and target variable.
def boxplot_percentage(data, target, categorical_var):
    # Create multi-index dataframe with primary index as the target, secondary index as categorical variable
    df_grouped = data.groupby([target, categorical_var])[categorical_var].count().to_frame()
    df_grouped = df_grouped.rename(columns = {categorical_var:'count'})

    # This code ensures that if a certain subcategory isn't present in the 'stroke' or 'no stroke' subset, 
    # it will be added, with a count of '0'
    df_grouped = df_grouped.unstack().fillna(0).stack().astype(int)

    # Add column which represents the categorical variable count as a percentage of target variable
    # (stroke vs. not stroke). I used range() to give them unique values for debugging purposes
    df_grouped['percent_of_target_cat'] = range(len(df_grouped))
    
    # Add column which represents the target variable count as a percentage of categorical variable
    df_grouped['percent_of_cat_var'] = range(len(df_grouped))

    # Loop through multi-index dataframe, giving the new columns they're appropriate values
    for target_value in df_grouped.index.levels[0]:
        for categorical_var_value in df_grouped.index.levels[1]:
            df_grouped.loc[(target_value, categorical_var_value), 'percent_of_target_cat'] = (df_grouped.loc[(target_value, categorical_var_value), 'count'] / df_grouped.loc[(target_value, slice(None)), :]['count'].sum()) * 100
            df_grouped.loc[(target_value, categorical_var_value), 'percent_of_cat_var'] = (df_grouped.loc[(target_value, categorical_var_value), 'count'] / df_grouped.loc[(slice(None), categorical_var_value), :]['count'].sum()) * 100

    # Convert from multi-index dataframe to two columns with those index values 
    # This will add columns for target and categorical variable value, as it makes it easier to create a boxplot
    df_grouped = df_grouped.reset_index()
    
    # Plots figure with target as x-axis labels, each with a bar for each categorical variable
    sns.barplot(x=df_grouped[target], y=df_grouped['percent_of_target_cat'], hue=df_grouped[categorical_var])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=categorical_var)
    plt.title('Percent ' + format_col(categorical_var) + ' by Stroke')
    plt.xlabel('Stroke')
    plt.ylabel('Percent ' + format_col(categorical_var))
    plt.show()
    
    # Plots figure with categorical variable as x-axis labels, each with a bar for each target variable
    sns.barplot(x=df_grouped[categorical_var], y=df_grouped['percent_of_cat_var'], hue=df_grouped[target])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=target)
    plt.title('Percent Stroke by ' + format_col(categorical_var))
    plt.xlabel(format_col(categorical_var))
    plt.ylabel('Percent Stroke')
    plt.show()
    
    # The above two figures are helpful in visualizing the complete picture, but contain some redundant information
    # Will simplify to just percent stroke in each category
    sns.barplot(x=df_grouped[categorical_var], y=df_grouped[(df_grouped['stroke']==1)]['percent_of_cat_var'])
    plt.title('Percent Stroke by ' + format_col(categorical_var))
    plt.xlabel(format_col(categorical_var))
    plt.ylabel('Percent Stroke')
    save_filename = 'perc_stroke_by_' + categorical_var
    save_image(output_dir, save_filename)
    plt.show()
    
for cat_cols in categorical_cols:
    boxplot_percentage(dataset, 'stroke', cat_cols)
    
#dataset[(dataset['stroke']==1)]['stroke']

# ==========================================================
# Continuous variables
# ==========================================================

# Numerical data histograms
for col in numerical_cols:
    #sns.distplot used to plot the histogram and fit line, but it's been deprecated to displot or histplot which don't 
    sns.distplot(dataset[col])
    plt.title(format_col(col) + ' Histogram')
    save_filename = 'hist_' + col
    save_image(output_dir, save_filename)
    plt.show()

# All continuous variables have somewhat normal or uniform distribution, 'avg_glucose_level' with an obvious positive skew and somewhat 
# bimodal and BMI with a slight positive skew

# Distribution plots separated by stroke
for col in numerical_cols:
    sns.kdeplot(data=dataset[dataset.stroke==1], x=col, shade=True, alpha=1, label='stroke')
    sns.kdeplot(data=dataset[dataset.stroke==0], x=col, shade=True, alpha=0.5, label='no stroke')
    plt.title(format_col(col) + ' Histogram by Stroke')
    save_filename = 'hist_by_stroke-' + col
    save_image(output_dir, save_filename)    
    plt.legend()
    plt.show()

# Continuous variables 'age' and 'avg_glucose_level' with observable difference in distribution in stroke vs. no stroke

# =============================
# Correlation between continuous variables
# =============================

# Find correlation between variables
# np.trui sets all the values above a certain diagonal to 0, so we don't have redundant boxes
matrix = np.triu(dataset[numerical_cols].corr()) 
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, mask=matrix, cmap="rocket", vmin=0, vmax=1)
plt.show()

# You can also make a correlation matrix that includes the diagonal so that the color spectrum better 
# represents the more extreme values
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title('Correlation Between Continuous Variables')
save_filename = 'correlation_cont_variables'
save_image(output_dir, save_filename)  
plt.show()

# Age has the highest correlation with other continuous variables

# Since age has the highest correlation with other variables, will plot those relationships
# First, scatterplots and lineplots showing relationship between age and other continuous variables
sns.scatterplot(data=dataset, x='age', y='bmi')
plt.show()

sns.lineplot(data=dataset, x='age', y='bmi')
plt.title('Relationship Between Age and BMI')
save_filename = 'correlation_age_bmi'
save_image(output_dir, save_filename)  
plt.show()

sns.scatterplot(data=dataset, x='age', y='avg_glucose_level')
plt.show()

sns.lineplot(data=dataset, x='age', y='avg_glucose_level')
plt.title('Relationship Between Age and Avg Glucose Level')
save_filename = 'correlation_age_avg_glucose_level'
save_image(output_dir, save_filename)  
plt.show()

# =============================
# Relationship between age and categorical variables (including stroke)
# =============================
# Second, boxplots showing relationship between age and categorical variables
for cat_col in categorical_cols:
    sns.boxplot(data=dataset, x=cat_col, y='age')
    plt.title('Relationship Between Age and ' + format_col(cat_col))
    save_filename = 'relationship_age_' + cat_col
    save_image(output_dir, save_filename)  
    plt.show()
    
sns.boxplot(data=dataset, x='stroke', y='age')
plt.title('Relationship Between Age and Stroke')
save_filename = 'relationship_age_stroke'
save_image(output_dir, save_filename)  
plt.show()

# Other than obvious relationships (higher age in those who were ever married), individuals with hypertension, heart disease, and stroke were older

# =============================
# Correlation between categorical variables
# =============================

# Could include chi-square test here

# =============================
# Cumulative Risk of stroke by age
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
plt.xlabel('Age')
plt.ylabel('Cumulative Stroke Risk')
plt.title('Cumulative Stroke Risk vs. Age')
save_filename = 'cumulative_stroke_risk_vs_age'
save_image(output_dir, save_filename)  
plt.show()

























