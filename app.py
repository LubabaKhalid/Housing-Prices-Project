import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('housing_prices.csv')

# Display dataset overview
st.title('Housing Price Prediction')
st.write("Dataset Overview")
st.dataframe(df.head())  # Display first few rows of the dataset

# Summary Statistics
st.write("Summary Statistics:")
st.write(df.describe())  # Summary statistics for numerical features

# Data Types and Unique Value Counts
st.write("Data Types and Unique Value Counts:")
st.write(df.dtypes)  # Show data types
st.write(df.nunique())  # Show unique value counts for each column

# Visualize Price Distribution (Histogram with KDE)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['price'], kde=True, bins=30, ax=ax)
st.pyplot(fig)  # Display price distribution

# Check for missing values
missing_data = df.isnull().sum()
st.write("Missing Values:")
st.write(missing_data)  # Show the count of missing values per column

# Visualize missing data as heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
st.pyplot(fig)  # Display the heatmap for missing values

# Perform one-hot encoding for categorical columns (convert to dummy variables)
df = pd.get_dummies(df, drop_first=True)

# Ensure all columns are numeric (for scaling and machine learning)
df = df.apply(pd.to_numeric, errors='coerce')

# Separate features (X) and target (y)
X = df.drop('price', axis=1)  # Features
y = df['price']               # Target

# Scaling the features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)  # Scale features excluding target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
# Make predictions on the test set
y_pred = model.predict(X_test)

# Model Evaluation: Displaying performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("Model Evaluation:")
st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

# Compute the residuals (errors)
residuals = y_test - y_pred

# Option 1: Residual plot with lowess smoothing (requires statsmodels)
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, ax=ax)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Error)')
    plt.title('Residual Plot (Lowess Smoothed)')
    st.pyplot(fig)
except RuntimeError:
    st.write("Error: `lowess=True` requires statsmodels. Switching to non-smoothed plot.")

    # Option 2: Residual plot without smoothing (no statsmodels needed)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.residplot(x=y_pred, y=residuals, ax=ax)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Error)')
    plt.title('Residual Plot (No Smoothing)')
    st.pyplot(fig)


# Visualize feature importance from the Random Forest model
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = X.columns[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=sorted_importance, y=sorted_features, ax=ax)
plt.title('Feature Importance')
st.pyplot(fig)

# Check for outliers using boxplot (e.g., for the 'price' feature)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['price'], ax=ax)
st.pyplot(fig)  # Display boxplot for price

# Explore the relationship between 'sqft_living' and 'price' with a scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df['sqft_living'], y=df['price'], ax=ax)
plt.title("Price vs Square Footage of Living Area")
plt.xlabel("Square Footage")
plt.ylabel("Price")
st.pyplot(fig)

# Correlation Heatmap for all numeric features
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plt.title("Correlation Heatmap")
st.pyplot(fig)

# Pairwise relationship of features related to price
sns.pairplot(df[['price', 'sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot']])
st.pyplot()

