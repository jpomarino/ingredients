import streamlit as st
from models import apriori

st.title("Ingredient Recommender")

model_choice = st.selectbox(
    "Choose a recommendation model:", ["Apriori", "Graph", "Word2Vec"]
)

ingredient = st.text_input("Enter an ingredient:")

if st.button("Recommend"):
    if model_choice == "Apriori":
        model = apriori.AprioriRecommender()
        rec = model.recommend(ingredient)
    #    elif model_choice == "Graph":
    #        rec = graph_model.recommend(ingredient)
    #    elif model_choice == "Word2Vec":
    #        rec = w2v_model.recommend(ingredient)

    st.write(rec)
