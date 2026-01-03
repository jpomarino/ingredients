# 🥑 Ingredient Recommender

A data-driven ingredient recommendation system that suggests ingredients that pair well together based on recipe data.
The project combines association rule mining, graph-based methods, and distributional semantics (Word2Vec), and is deployed as an interactive Streamlit web app.

Try the deployed app here:  
👉 https://ingredients-agqz9rrstjm7vblaztp9oz.streamlit.app/

## 🚀 Project Overview

Given an ingredient (e.g. "apricot"), the app recommends other ingredients that commonly co-occur with it in recipes using one of three models:

1. Apriori (Association Rules)

    * Uses frequent pattern mining and association rules
    * Recommendations ranked by lift

2. Graph-based Model (Ingredient Co-occurrence Network)

    * Ingredients represented as nodes
    * Edge weights based on co-occurrence strength
    * Recommendations derived from graph neighborhoods / pointwise mutual information

3. Word2Vec (Ingredient Embeddings)

    * Treats recipes as “sentences” and ingredients as “words”
    * Learns dense vector representations
    * Recommendations based on cosine similarity in the embedding space

The system is designed for low-latency inference, with all heavy preprocessing done offline.

## 🧠 Motivation

I love cooking, and sometimes I need a quick intuition for what ingredients I can use with what I have in the firdge. I found a Kaggle dataset (https://www.kaggle.com/datasets/alincijov/cooking-ingredients?select=train.csv) of recipes and the ingredients they used. I decided to use this dataset to create ingredient recommendations using three different models.

Through this project, I got hands-on experience in recommendation models based on co-occurrence, and I used Streamlit for the first time to deploy them!
