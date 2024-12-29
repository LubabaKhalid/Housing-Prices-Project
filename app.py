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
st.title('Enhanced Housing Price Prediction Analysis')
st.write("Dataset Overview")
st.dataframe(df.head())  # Display first few rows of the dataset

# Summary Statistics
st.write("Summary Statistics:")
st.write(df.describe())  # Summary statistics for numerical features

# Data Types and Unique Value Counts
st.write("Data Types and Unique Value Counts:")
st.write(df.dtypes)  # Show data types
st.write(df.nunique())  # Show unique value counts for each column

# 1. Visualize Price Distribution (Histogram with KDE)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['price'], kde=True, bins=30, ax=ax)
st.pyplot(fig)  # Display price distribution

# 2. Check for missing values
missing_data = df.isnull().sum()
st.write("Missing Values:")
st.write(missing_data)  # Show the count of missing values per column

# 3. Visualize missing data as heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
st.pyplot(fig)  # Display the heatmap for missing values

# 4. Perform one-hot encoding for categorical columns (convert to dummy variables)
df = pd.get_dummies(df, drop_first=True)

# 5. Ensure all columns are numeric (for scaling and machine learning)
df = df.apply(pd.to_numeric, errors='coerce')

# 6. Correlation Heatmap for all numeric features
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# 7. Pairwise relationships using seaborn pairplot
pairplot_fig = sns.pairplot(df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']])
st.pyplot(pairplot_fig)

# 8. Price Distribution by Number of Bedrooms (Boxplot)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['bedrooms'], y=df['price'], ax=ax)
ax.set_title("Price Distribution by Bedrooms")
st.pyplot(fig)

# 9. Relationship between 'area' and 'price' with a scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df['area'], y=df['price'], ax=ax)
ax.set_title("Price vs Area")
ax.set_xlabel("Area")
ax.set_ylabel("Price")
st.pyplot(fig)

# 10. Distribution of Parking availability (with respect to price)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['parking'], y=df['price'], ax=ax)
ax.set_title("Price Distribution by Parking Availability")
st.pyplot(fig)

# 11. Relationship between 'stories' and 'price'
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['stories'], y=df['price'], ax=ax)
ax.set_title("Price Distribution by Number of Stories")
st.pyplot(fig)

# 12. Pairplot for bedrooms and bathrooms relationship to price
pairplot_bed_bath = sns.pairplot(df[['price', 'bedrooms', 'bathrooms']])
st.pyplot(pairplot_bed_bath)

# 13. Visualize feature importance from Random Forest model
X = df.drop('price', axis=1)  # Features
y = df['price']  # Target

# Scaling the features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(scaled_features, y)

# Feature Importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = X.columns[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=sorted_importance, y=sorted_features, ax=ax)
plt.title('Feature Importance from Random Forest Model')
st.pyplot(fig)

# 14. Explore distribution of 'furnishingstatus' columns
furnishing_status = df[['furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']].sum()
st.write("Furnishing Status Distribution:")
st.write(furnishing_status)

# 15. Price Distribution by Furnishing Status (Boxplot)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['furnishingstatus_semi-furnished'], y=df['price'], ax=ax)
ax.set_title("Price Distribution by Semi-furnished Status")
st.pyplot(fig)

# Train-test split, Random Forest Model Evaluation (MAE, MSE, R^2)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model.fit(X_train, y_train)

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
