
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