import streamlit as st
import pandas as pd
from models import apriori, graph, word2vec

# Configure page
st.set_page_config(
    page_title="Ingredient Recommender", page_icon="🥗", layout="centered"
)
st.title("🥗 Ingredient Recommender")
st.caption("Discover ingredients that pair well together")

TOP_K = 20


# Load models and cache them
@st.cache_resource
def load_models():
    return {
        "Apriori": apriori.AprioriRecommender(),
        "Graph": graph.GraphRecommender(),
        "Word2Vec": word2vec.Word2VecRecommender(),
    }


models = load_models()

# Model selection
model_choice = st.selectbox("Choose a recommendation model:", list(models.keys()))

model = models[model_choice]

# Ingredient selection
ingredient = st.selectbox(
    "Choose an ingredient:",
    sorted(model.supported_ingredients),
    key=f"{model_choice}_ingredient_select",
)

# Score label per model
SCORE_NAME = {"Apriori": "Lift", "Graph": "PMI", "Word2Vec": "Cosine Similarity"}[
    model_choice
]

# Recommend button
if st.button("Recommend", type="primary"):
    rec = model.recommend(ingredient, top_k=TOP_K)

    if not rec:
        st.warning("No recommendations found.")
    else:
        # Convert dict → DataFrame
        df_rec = (
            pd.DataFrame(rec.items(), columns=["Ingredient", SCORE_NAME])
            .sort_values(SCORE_NAME, ascending=False)
            .reset_index(drop=True)
        )

        st.subheader(f"Top {len(df_rec)} Recommendations")
        st.dataframe(df_rec, use_container_width=True)

        # Optional bar chart
        st.bar_chart(
            df_rec.set_index("Ingredient")[SCORE_NAME],
            horizontal=True,
            sort=f"-{SCORE_NAME}",
        )

# Footer
st.markdown("---")
st.caption(
    "Models: Apriori (association rules), "
    "Graph-based co-occurrence, "
    "Word2Vec embeddings"
)

# Side bar
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by **Jose Pomarino Nima**  \nIngredient Recommendation System"
)
