
"""
Problem Statement :- 

The objective of this task is to clean and preprocess the Titanic dataset to make it suitable for further analysis and machine learning. The dataset contains passenger information such as age, gender, class, and survival status, but it includes missing values, inconsistent data types, and unnecessary columns. The goal is to prepare a clean and structured dataset.

Dataset Details:- 
    Dataset Name: Titanic Dataset
    Source: Kaggle (Titanic - Machine Learning from Disaster)
    Total Records: 891 rows
    Total Features: 12 columns


Approach (Step-by-Step):- 
    1. Data Loading
        Imported dataset using pandas
        Displayed first few rows using head()
        Checked structure using info()
    2. Handling Missing Values
        Filled missing values in Age using median
        Filled missing values in Embarked using mode
        Dropped Cabin column due to high missing values
    3. Removing Duplicates
        Checked for duplicate rows
        Removed duplicates using drop_duplicates()
    4. Data Type Conversion
        Converted Age column to integer type for consistency
    5. Column Renaming
        Renamed columns for better readability:
        Pclass → Passenger_Class
        SibSp → Siblings_Spouses
        Parch → Parents_Children
    6. Feature Selection
        Removed unnecessary columns like:
        Name
    Ticket
    7. Final Output
        Cleaned dataset saved as:
"""


# TASK 1: Titanic Dataset -DATA CLEANING


import pandas as pd
import numpy as np

# 1. Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")   # change path if needed

# 2. Basic Inspection
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Remove Duplicates
df.drop_duplicates(inplace=True)

# 4. Handle Missing Values

# Age → fill with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Embarked → fill with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Cabin → drop (too many missing values)
df.drop(columns=["Cabin"], inplace=True)

# 5. Convert Data Types (if needed)
# To safely convert from float64 to nullable Int64, first cast to standard int.
df["Age"] = df["Age"].astype(int).astype("Int64")

# 6. Rename Columns (clean format)
df.rename(columns={
    "Pclass": "Passenger_Class",
    "SibSp": "Siblings_Spouses",
    "Parch": "Parents_Children"
}, inplace=True)

# 7. Drop Unnecessary Columns (optional but good practice)
df.drop(columns=["Name", "Ticket"], inplace=True)

# 8. Final Check
print("\nCleaned Dataset Info:")
print(df.info())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Dataset Preview:")
print(df.head())

# 9. Save Cleaned Dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as cleaned_titanic.csv")