import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Streamlit app title
st.title("Car Price Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Import Data", "Pattern Visualization", "Descriptive Statistics", "Grouping", "Correlation and Causation"])

# Load the data
@st.cache_data
def load_data():
    path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    df = pd.read_csv(path)
    return df

df = load_data()

# Section 1: Import Data
if section == "Import Data":
    st.header("1. Import Data")
    st.write("Data loaded successfully!")
    st.write(df.head())

# Section 2: Pattern Visualization
elif section == "Pattern Visualization":
    st.header("2. Analyzing Individual Feature Patterns Using Visualization")
    
    # Display data types
    st.write("Data types of each column:")
    st.write(df.dtypes)
    
    # Question 1: Data type of 'peak-rpm'
    st.write("**Question 1:** The data type of the column 'peak-rpm' is:", df['peak-rpm'].dtype)
    
    # Correlation example
    st.write("Correlation between 'diesel' and 'price':")
    st.write(df[['diesel', 'price']].corr())
    
    # Question 2: Correlation between 'bore', 'stroke', 'compression-ratio', 'horsepower'
    st.write("**Question 2:** Correlation between 'bore', 'stroke', 'compression-ratio', and 'horsepower':")
    correlation_matrix = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
    st.write(correlation_matrix)
    
    # Continuous Numerical Variables
    st.subheader("Continuous Numerical Variables")
    
    # Positive Linear Relationship: Engine-size vs Price
    st.write("Positive Linear Relationship: Engine-size vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x="engine-size", y="price", data=df, ax=ax)
    ax.set_xlabel("Engine Size")
    ax.set_ylabel("Price")
    ax.set_ylim(0)
    st.pyplot(fig)
    
    # Highway-mpg vs Price
    st.write("Negative Linear Relationship: Highway-mpg vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x="highway-mpg", y="price", data=df, ax=ax)
    ax.set_xlabel("Highway MPG")
    ax.set_ylabel("Price")
    st.pyplot(fig)
    
    # Weak Linear Relationship: Peak-rpm vs Price
    st.write("Weak Linear Relationship: Peak-rpm vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x="peak-rpm", y="price", data=df, ax=ax)
    ax.set_xlabel("Peak RPM")
    ax.set_ylabel("Price")
    st.pyplot(fig)
    
    # Question 3a: Correlation between 'stroke' and 'price'
    st.write("**Question 3a:** Correlation between 'stroke' and 'price':")
    st.write(df[['stroke', 'price']].corr())
    
    # Question 3b: Regplot for 'stroke' vs 'price'
    st.write("**Question 3b:** Regplot for 'stroke' vs 'price':")
    fig, ax = plt.subplots()
    sns.regplot(x="stroke", y="price", data=df, ax=ax)
    st.pyplot(fig)
    
    # Categorical Variables
    st.subheader("Categorical Variables")
    
    # Boxplot: Body-style vs Price
    st.write("Boxplot: Body-style vs Price")
    fig, ax = plt.subplots()
    sns.boxplot(x="body-style", y="price", data=df, ax=ax)
    ax.set_xlabel("Body style")
    ax.set_ylabel("Price")
    st.pyplot(fig)
    
    # Boxplot: Engine-location vs Price
    st.write("Boxplot: Engine-location vs Price")
    fig, ax = plt.subplots()
    sns.boxplot(x="engine-location", y="price", data=df, ax=ax)
    ax.set_xlabel("Engine location")
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)
    
    # Boxplot: Drive-wheels vs Price
    st.write("Boxplot: Drive-wheels vs Price")
    fig, ax = plt.subplots()
    sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax)
    ax.set_xlabel("Drive Wheels")
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)

# Section 3: Descriptive Statistics
elif section == "Descriptive Statistics":
    st.header("3. Descriptive Statistical Analysis")
    
    # Describe continuous variables
    st.write("Descriptive statistics for continuous variables:")
    st.write(df.describe())
    
    # Describe categorical variables
    st.write("Descriptive statistics for categorical variables:")
    st.write(df.describe(include=['object']))
    
    # Value Counts for 'drive-wheels'
    st.write("Value counts for 'drive-wheels':")
    drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
    drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
    drive_wheels_counts.index.name = 'drive-wheels'
    st.write(drive_wheels_counts)
    
    # Value Counts for 'engine-location'
    st.write("Value counts for 'engine-location':")
    engine_loc_counts = df['engine-location'].value_counts().to_frame()
    engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
    engine_loc_counts.index.name = 'engine-location'
    st.write(engine_loc_counts)

# Section 4: Grouping
elif section == "Grouping":
    st.header("4. Basics of Grouping")
    
    # Group by 'drive-wheels' and 'body-style'
    st.write("Group by 'drive-wheels' and 'body-style':")
    df_group_one = df[['drive-wheels', 'body-style', 'price']]
    df_group_one = df_group_one.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    st.write(df_group_one)
    
    # Pivot table
    st.write("Pivot table for 'drive-wheels' and 'body-style':")
    grouped_pivot = df_group_one.pivot(index='drive-wheels', columns='body-style')
    grouped_pivot = grouped_pivot.fillna(0)
    st.write(grouped_pivot)
    
    # Heatmap
    st.write("Heatmap for 'drive-wheels' and 'body-style' vs Price:")
    fig, ax = plt.subplots()
    im = ax.pcolor(grouped_pivot, cmap='RdBu')
    ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(grouped_pivot.columns.levels[1], minor=False)
    ax.set_yticklabels(grouped_pivot.index, minor=False)
    plt.xticks(rotation=90)
    fig.colorbar(im)
    st.pyplot(fig)

# Section 5: Correlation and Causation
elif section == "Correlation and Causation":
    st.header("5. Correlation and Causation")
    
    # Pearson Correlation examples
    st.write("Pearson Correlation examples:")
    
    # Wheel-Base vs Price
    st.write("Wheel-Base vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Horsepower vs Price
    st.write("Horsepower vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Length vs Price
    st.write("Length vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Width vs Price
    st.write("Width vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Curb-Weight vs Price
    st.write("Curb-Weight vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Engine-Size vs Price
    st.write("Engine-Size vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Bore vs Price
    st.write("Bore vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # City-mpg vs Price
    st.write("City-mpg vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")
    
    # Highway-mpg vs Price
    st.write("Highway-mpg vs Price:")
    pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef}, P-value: {p_value}")