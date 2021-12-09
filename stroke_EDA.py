import pandas as pd
import numpy as np
import scipy.stats as ss
from os import chdir
from pathlib import PureWindowsPath, Path
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.impute import SimpleImputer


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

# Will create list of categorical variables with target and one without target
cat_cols_w_target = categorical_cols.copy()
categorical_cols.remove('stroke')

# Map 'hypertension' and 'heart_disease' numeric values to categorical values
hypertension_dict = {0:'normotensive', 1:'hypertensive'}
heart_disease_dict = {0:'no_heart_disease', 1:'heart_disease'}
dataset['hypertension'] = dataset['hypertension'].map(hypertension_dict)
dataset['heart_disease'] = dataset['heart_disease'].map(heart_disease_dict)

# =======================================================================================
# Visualize data
# =======================================================================================

# Standardize image saving parameters
def save_image(dir, filename, dpi=300, bbox_inches='tight'):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches)

# Create dictionary of formatted column names  to be used for
# figure labels (title() capitalizes every word in a string)
formatted_cols = {}
for col in dataset.columns:
    formatted_cols[col] = col.replace('_', ' ').title()
formatted_cols['bmi'] = 'BMI'

# Function returning the formatted version of column name
def format_col(col_name):
    return formatted_cols[col_name]

# Create 2d array of given size, used for figures with gridspec
def create_2d_array(num_rows, num_cols):
    matrix = []
    for r in range(0, num_rows):
        matrix.append([0 for c in range(0, num_cols)])
    return matrix

# ==========================================================
# Categorical variables
# ==========================================================

# Categorical data bar charts, total count of each category
for col in cat_cols_w_target:
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts())
    plt.title(format_col(col) + ' Count')
    plt.ylabel('Count')
    plt.xlabel(format_col(col))
    save_filename = 'counts_' + col
    save_image(output_dir, save_filename, bbox_inches='tight')
    plt.show()
    
# Very few cases of stroke, hypertension, and heart disease. It will be interesting to see if they are correlated.

   
# =============================
# Combine into one figure
# =============================
# Create figure, gridspec, and 2d array of axes/subplots with given number of rows and columns
fig = plt.figure(constrained_layout=True, figsize=(16, 8))
num_rows = 2
num_cols = 4
ax_array = create_2d_array(num_rows, num_cols)
gs = fig.add_gridspec(num_rows, num_cols)

# Map each subplot/axis to gridspec location
for r in range(len(ax_array)):
    for c in range(len(ax_array[r])):
        ax_array[r][c] = fig.add_subplot(gs[r,c])

# Flatten 2d array of axis objects to iterate through easier
ax_array_flat = np.array(ax_array).flatten()

# Loop through categorical variables, plotting each in the figure
i = 0
for col in cat_cols_w_target:
    axis = ax_array_flat[i]
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts(), ax=axis)
    axis.set_title(format_col(col) + ' Count')
    axis.set_xlabel(format_col(col))
    
    # Rotate x-axis tick labels so they don't overlap
    plt.setp(axis.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    # Only want to label the y-axis on the first subplot of each row
    if i % 4 == 0:
        axis.set_ylabel('Count')
    else:
        # set visibility of y-axis as False
        axis.set_ylabel('')
    i += 1

# Finalize figure formatting and export
fig.suptitle('Categorical Variable Counts', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
save_filename = 'combined_cat_counts'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()


# =============================
# Relationship between categorical data and target
# =============================
# These bar charts use the categorical variable on the x-axis, each one split by stroke vs. no stroke
for col in categorical_cols:
    sns.catplot(data=dataset, x=col, hue="stroke", kind="count", legend=False)
    plt.title('Stroke Count by ' + format_col(col))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='stroke')
    save_filename = 'count_stroke_by_' + col
    save_image(output_dir, save_filename)
    plt.show()
# Not very helpful visualization as the number of records with stroke is very low

# This version has stroke on the x-axis, each one split by how many records are in each categorical variable
for col in categorical_cols:
    sns.catplot(data=dataset, x="stroke", hue=col, kind="count")
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
    plt.title(format_col(col) + ' Distribution by Outcome')
    save_filename = 'hist_by_stroke-' + col
    plt.legend()
    save_image(output_dir, save_filename)    
    plt.show()

# Continuous variables 'age' and 'avg_glucose_level' with observable difference in distribution in stroke vs. no stroke


# =======================================================================================
# Correlation between variables
# =======================================================================================
# ==========================================================
# Correlation between continuous variables
# ==========================================================

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

# =============================
# Further exploration correlation continuous variables
# =============================
# Since age has the highest correlation with other variables, will plot those relationships
# Scatterplots and lineplots showing relationship between age and other continuous variables
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

# ==========================================================
# Association between categorical variables
# ==========================================================
# Credit to: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

# Calculate Cramér’s V (based on a nominal variation of Pearson’s Chi-Square Test) between two categorical featuers 'x' and 'y'
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# New dataframe to store results for each combination of categorical variables
cramers_df = pd.DataFrame(columns=cat_cols_w_target, index=cat_cols_w_target)

# Loop through each paring of categorical variables, calculating the Cramer's V for each and storing in dataframe
for col in cramers_df.columns:
    for row in cramers_df.index:
        cramers_df.loc[[row], [col]] = cramers_v(dataset[row], dataset[col])

# Values default to 'object' dtype, will convert to numeric
cramers_df = cramers_df.apply(pd.to_numeric)

# Output results as heatmap
sns.heatmap(cramers_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title("Association Between Categorical Variables (Cramér's V)")
save_filename = 'correlation_cat_variables'
save_image(output_dir, save_filename)  
plt.show()

# =============================
# Further exploration association categorical variables
# =============================
# Plot catplots of categorical variables with correlation ratio > 0.29
# Loop through cramers_df diagonally to skip redundant pairings
for col in range(len(cramers_df.columns)-1):
    for row in range(col+1, 8):
        cramers_value = cramers_df.iloc[[row], [col]].iat[0,0].round(2)
        if cramers_value > 0.29:
            column_name = cramers_df.columns[col]
            row_name = cramers_df.index[row]
            sns.catplot(data=dataset, x=column_name, hue=row_name, kind="count", legend=False)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=row_name)
            plt.title(format_col(column_name) + ' vs. ' + format_col(row_name) + " (Cramer's=" + str(cramers_value) + ')')
            save_filename = 'compare_' + column_name + '_vs_' + row_name
            save_image(output_dir, save_filename)
            plt.show()

# ==========================================================
# Correlation between continuous and categorical variables
# ==========================================================
# Credit to: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

# Calculate correlation ratio between a categorical feature ('categories') and numeric feature ('measurements')
def correlation_ratio(categories, measurements):
    merged_df = categories.to_frame().merge(measurements, left_index=True, right_index=True)
    fcat, _ = pd.factorize(categories)
    merged_df['fcat'] = fcat
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = merged_df[merged_df['fcat']==i][measurements.name]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

# Grab numerical data from original dataset so that I can impute values(correlation_ratio() cannot handle unknown values)
num_df = dataset[numerical_cols]
num_df = pd.DataFrame(SimpleImputer().fit_transform(num_df), columns=numerical_cols, index=dataset.index)

# New dataframe to store results for each combination of numerical and categorical variables
corr_ratio_df = pd.DataFrame(columns=num_df.columns, index=cat_cols_w_target)

# Loop through each paring of numerical and categorical variables, calculating the correlation ratio for each and storing in dataframe
for col in corr_ratio_df.columns:
    for row in corr_ratio_df.index:
        corr_ratio_df.loc[[row], [col]] = correlation_ratio(dataset[row], num_df[col])

# Values default to 'object' dtype, will convert to numeric
corr_ratio_df = corr_ratio_df.apply(pd.to_numeric)

# Output results as heatmap
sns.heatmap(corr_ratio_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title("Correlation Ratio Between Numerical and Categorical Variables")
save_filename = 'correlation_cat_num_variables'
save_image(output_dir, save_filename)  
plt.show()

# =============================
# Further exploration correlation continuous and categorical variables
# =============================
# Plot boxplots of  continuous and categorical variables with correlation ratio > 0.3
for col in corr_ratio_df.columns:
    for row in corr_ratio_df.index:
        corr_value = corr_ratio_df.loc[[row], [col]].iat[0,0].round(2)
        if corr_value > 0.3:
            sns.boxplot(data=dataset, x=row, y=col)
            plt.title(format_col(col) + ' vs. ' + format_col(row) + ' (Corr Ratio=' + str(corr_value) + ')')
            save_filename = 'relationship_' + col + '_' + row
            save_image(output_dir, save_filename)  
            plt.show()

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

























