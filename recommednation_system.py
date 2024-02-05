# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:08:07 2024

@author: AAKASH
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For saving the model
import streamlit as st

# Function to recommend hotels based on country name
def recommend_hotels(country_name):
    # Load saved model
    vectorizer = joblib.load('vectorizer.pkl')
    feature_vectors = joblib.load('feature_vectors.pkl')

    # Filter hotels by country name
    filtered_hotels = hotel_df[hotel_df['country'].str.lower() == country_name]

    # Check if any hotels exist for the given country
    if not filtered_hotels.empty:
        # Get frequency of each hotel
        hotel_counts = filtered_hotels['hotelname'].value_counts()

        # Select top hotels based on frequency
        top_hotels = hotel_counts.head(25)  # Adjust the number as needed

        return top_hotels.index.tolist()
    else:
        return None

# Load the hotel data
hotel_df1 = pd.read_csv('C:/Users/AAKASH/Desktop/Recommendation_system/Hotel_details.csv')
hotel_df = hotel_df1.head(10000)

# Fill missing values
req_features = ['hotelname', 'city', 'country', 'propertytype', 'starrating']
for i in req_features:
    hotel_df[i] = hotel_df[i].fillna('')

# Combine features
comb_feature = hotel_df['hotelname'].astype(str) + ' ' + hotel_df[
    'city'].astype(str) + ' ' + hotel_df['country'].astype(str) + ' ' + hotel_df[
                   'propertytype'].astype(str) + ' ' + hotel_df['starrating'].astype(str)

# Text to feature data
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(comb_feature)

# Save the model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(feature_vectors, 'feature_vectors.pkl')

# Streamlit web application
st.title('Hotel Recommendation System')

# User input for country name
country_name = st.text_input("Enter the country name: ").strip().lower()  # or .upper()

# Filter DataFrame by country name (case insensitive)
filtered_hotels = hotel_df[hotel_df['country'].str.lower() == country_name]

# Check if any hotels exist for the given country
if not filtered_hotels.empty:
    # Get frequency of each hotel
    hotel_counts = filtered_hotels['hotelname'].value_counts()

    # Select top 20 to 25 hotels based on frequency
    top_hotels = hotel_counts.head(25)  # Adjust the number as needed

    # Display the top hotels
    st.write(f"Top 20 to 25 hotels in {country_name.capitalize()}:")  # Capitalize the first letter for display
    for hotel, count in top_hotels.items():
        st.write(f"{hotel}")
else:
    st.write(f"No hotels found in {country_name.capitalize()}")
