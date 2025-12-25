# ingredients

🥑 Ingredient Recommendation System

A data-driven ingredient recommendation system that suggests ingredients that pair well together based on recipe data.
The project combines association rule mining, graph-based methods, and distributional semantics (Word2Vec), and is deployed as an interactive Streamlit web app.

🚀 Project Overview

Given an ingredient (e.g. apricot), the app recommends other ingredients that commonly co-occur with it in recipes using one of three models:

Apriori (Association Rules)

Uses frequent pattern mining and association rules

Recommendations ranked by lift

Graph-based Model (Ingredient Co-occurrence Network)

Ingredients represented as nodes

Edge weights based on co-occurrence strength

Recommendations derived from graph neighborhoods / scores

Word2Vec (Ingredient Embeddings)

Treats recipes as “sentences” and ingredients as “words”

Learns dense vector representations

Recommendations based on cosine similarity

The system is designed for low-latency inference, with all heavy preprocessing done offline.

🧠 Motivation

Ingredient pairing is a classic recommendation problem with real-world applications:

Recipe discovery

Meal planning

Grocery assistance

Culinary creativity

This project explores how different modeling paradigms (rules, graphs, embeddings) behave on the same problem — and exposes them through a unified interface.
