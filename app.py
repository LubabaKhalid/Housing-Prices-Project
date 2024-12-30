import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit.components.v1 as components

# Set background image
background_image_path = r"C:\Users\PMLS\Desktop\Housing-Prices-Project\images\house.jpg"
background_html = """
    <style>
    body {
        background-image: url("{background_image_path}");  /* Relative path to the image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #000000;  /* Ensure text color is black */
    }

    #main {
        background: rgba(255, 255, 255, 0.8);  /* White background with 80% opacity for content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Adding a slight shadow to make the content area stand out */
    }

    h1, h2, h3, h4, p {
        color: #003366;  /* Set text color for headings and paragraphs */
    }
    </style>
    """
# Inject custom HTML to apply the background image
components.html(background_html, height=0)

# Load the dataset
df = pd.read_csv('housing_prices.csv')

# Custom Title and Styling
st.markdown("<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Housing Price Prediction Analysis</h1>", unsafe_allow_html=True)

# Add a main container for content to ensure background overlay
st.markdown('<div id="main">', unsafe_allow_html=True)

# Create a sidebar for buttons and interactions
st.sidebar.title("Navigation")
button_overview = st.sidebar.button("Show Dataset Overview")
button_summary = st.sidebar.button("Show Summary Statistics")
button_visualize = st.sidebar.button("Show Visualizations")
button_model = st.sidebar.button("Show Model Evaluation")

# Display Dataset Overview Section
if button_overview:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Dataset Overview</h4>", unsafe_allow_html=True)
    st.dataframe(df.head())  # Display first few rows of the dataset

# Display Summary Statistics Section
if button_summary:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Summary Statistics</h4>", unsafe_allow_html=True)
    st.write(df.describe())  # Summary statistics for numerical features

# Create columns for better visualizations layout
col1, col2 = st.columns(2)

# Display Visualizations Section
if button_visualize:
    # Data Types and Unique Value Counts
    with col1:
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Data Types and Unique Value Counts</h4>", unsafe_allow_html=True)
        st.write(df.dtypes)  # Show data types

    with col2:
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Unique Value Counts for Each Column</h4>", unsafe_allow_html=True)
        st.write(df.nunique())

    # Price Distribution
    with col1:
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Price Distribution (Histogram with KDE)</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['price'], kde=True, bins=30, ax=ax, color='skyblue')
        st.pyplot(fig)

    with col2:
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Missing Values Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

    categorical_cols = df.select_dtypes(include=['object']).columns

# Use .map() to convert categorical columns like 'yes'/'no' to numeric values
    for col in categorical_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(df[col])

    # Perform one-hot encoding for any remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Now calculate and plot the correlation heatmap
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Correlation Heatmap</h4>", unsafe_allow_html=True)

    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Feature Correlation", fontsize=22, fontweight='bold')

    # Display the plot in Streamlit
    st.pyplot(fig)
    

    # Pairplot
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Pairplot of Selected Features</h4>", unsafe_allow_html=True)
    pairplot_fig = sns.pairplot(df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']], palette='coolwarm')
    st.pyplot(pairplot_fig)
    
    # Boxplot of Price Distribution by Number of Bedrooms
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Price Distribution by Number of Bedrooms</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df['bedrooms'], y=df['price'], ax=ax, palette='Set2')
    ax.set_title("Price Distribution by Bedrooms", fontsize=22, fontweight='bold')
    st.pyplot(fig)

# Model Evaluation Section
if button_model:
    # Perform one-hot encoding for categorical columns (convert to dummy variables)
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all columns are numeric (for scaling and machine learning)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Prepare features and target
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

    # Feature Importance Plot
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Feature Importance</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importance, y=sorted_features, ax=ax, palette='Blues')
    plt.title('Feature Importance from Random Forest Model', fontsize=22, fontweight='bold')
    st.pyplot(fig)

    # Train-Test Split Evaluation
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Train-Test Split Evaluation</h4>", unsafe_allow_html=True)

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

    st.write(f'Mean Absolute Error: {mae:.2f}')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R-squared: {r2:.2f}')

# End of content section
st.markdown('</div>', unsafe_allow_html=True)

