#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data from the CSV file into a pandas dataframe
movies_data = pd.read_csv("movies.csv")

# Streamlit UI
st.title("Movie Recommendation System")

# Sidebar for user input
st.sidebar.title("User Input")
movie_name = st.sidebar.text_input('Enter your favorite movie name:', 'Inception')

if st.sidebar.button('Get Recommendations'):
    # Selecting relevant features for recommendation
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Replace null values with an empty string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combine selected features
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                        movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

    # Convert text data to feature vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Calculate similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)

    # Creating a list with all the movie names given in the dataset
    list_of_all_titles = movies_data['title'].tolist()

    # Finding the close match for the movie name given by the user
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]

    # Finding the index of the movie with title
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # Getting a list of similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Print the name of similar movies based on the index
    st.header('Movies suggested for you:')
    recommended_movies = []
    for movie in sorted_similar_movies[1:31]:  # Exclude the first one (it's the same movie as the input)
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)

    st.write(recommended_movies)

