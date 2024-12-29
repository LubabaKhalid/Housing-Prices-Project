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
