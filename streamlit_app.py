import streamlit as st
from models import apriori, graph

top_k = 20

st.title("Ingredient Recommender")

model_choice = st.selectbox(
    "Choose a recommendation model:", ["Apriori", "Graph", "Word2Vec"]
)

ingredient = st.text_input("Enter an ingredient:")

if st.button("Recommend"):
    if model_choice == "Apriori":
        model = apriori.AprioriRecommender()
        rec = model.recommend(ingredient, top_k=top_k)
    elif model_choice == "Graph":
        model = graph.GraphRecommender()
        rec = model.recommend(ingredient, top_k=top_k)
    #    elif model_choice == "Word2Vec":
    #        rec = w2v_model.recommend(ingredient)

    st.write(rec)
