import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('housing_prices.csv')

# Display dataset
st.title('Housing Price Prediction')
st.write("Dataset Overview")
st.dataframe(df.head())

st.write("Summary Statistics:")
st.write(df.describe())

st.write("Data Types and Unique Value Counts:")
st.write(df.dtypes)
st.write(df.nunique())




st.write("Price Distribution:")
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True, bins=30)
st.pyplot()

# Check for missing values
missing_data = df.isnull().sum()
st.write("Missing Values:", missing_data)

# Fill missing values (for simplicity, using mean)
df.fillna(df.mean(), inplace=True)


df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('price', axis=1))  # Scale features excluding the target


X = scaled_features
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
