import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
import base64
import os
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_data  # Import data loader

# --- Streamlit Page Config ---
st.set_page_config(page_title="Thought out recommendations", layout="centered")



# Get the script's directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Function to Convert Image to Base64 ---
def get_base64_encoded_image(image_filename):
    """Load an image and encode it in Base64."""
    image_path = os.path.join(BASE_DIR, image_filename)  # Construct relative path
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Image file {image_path} not found.")
        return None  # Handle missing file gracefully
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Background Image Encoding ---
background_image_base64 = get_base64_encoded_image("img.jpeg")

# --- Inject CSS for Background Image and Centering ---
page_bg = f"""
<style>
    body {{
        background: url("data:image/jpeg;base64,{background_image_base64}") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Times New Roman', serif;
        text-align: center;
    }}

    .stApp {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }}

    h1, h2, h3, h4, h5, h6, p, div, input, textarea, label, button {{
        font-family: 'Times New Roman', serif !important;
    }}

    .stTextInput > div > div > input {{
        text-align: center;
        font-family: 'Times New Roman', serif !important;
    }}

    .stButton > button {{
        font-family: 'Times New Roman', serif !important;
    }}

    .recommendation-card {{
        padding: 15px; 
        border-radius: 10px; 
        background-color: rgba(255, 255, 255, 0.85);
        margin-bottom: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        font-family: 'Times New Roman', serif;
        text-align: center;
    }}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- Title ---
st.title(" Thought out Recommendations ")

# ----------------------------------------------------
# 1. Load Model and Data using `data_loader.py`
# ----------------------------------------------------

# --- Define Hybrid Model Class ---
class HybridPreferenceModel(torch.nn.Module):
    def __init__(self, n_users, n_items, user_feat_dim, embedding_dim=16):
        super().__init__()
        self.cf_user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.cf_item_embedding = torch.nn.Embedding(n_items, embedding_dim)
        self.user_mlp = torch.nn.Sequential(
            torch.nn.Linear(user_feat_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, embedding_dim)
        )

    def forward(self, user_ids, item_ids, user_features=None):
        cf_user = self.cf_user_embedding(user_ids)
        cf_item = self.cf_item_embedding(item_ids)
        content_user = self.user_mlp(user_features) if user_features is not None else 0
        final_user = cf_user + content_user
        scores = (final_user * cf_item).sum(dim=1)
        return scores


@st.cache_resource
def load_model():
    """Load trained hybrid recommendation model using a relative path."""
    model_path = os.path.join(BASE_DIR, "hybrid_model.pt")  # Correct relative path
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

@st.cache_resource
def load_mappings():
    """Load user & product mappings using a relative path."""
    mappings_path = os.path.join(BASE_DIR, "mappings.pkl")  # Correct relative path
    with open(mappings_path, "rb") as f:
        return pickle.load(f)

hybrid_model = load_model()
mappings = load_mappings()
user2idx = mappings["user2idx"]
idx2item = mappings["idx2item"]

# Load preprocessed data
data_frames = load_data()
clients_df = data_frames["clients"]
products_df = data_frames["products"]
stocks_df = data_frames["stocks"]
transactions_df = data_frames["transactions"]

# ----------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------

def get_family_level2(product_id):
    """Returns FamilyLevel2 from products.csv based on ProductID."""
    row = products_df[products_df["ProductID"] == product_id]
    return row["FamilyLevel2"].values[0] if not row.empty else "Unknown Product"

def check_stock(product_id):
    """Check if a product is in stock based on stocks.csv."""
    row = stocks_df[stocks_df["ProductID"] == product_id]
    return "✅ In Stock" if not row.empty and row["Quantity"].values[0] > 0 else "❌ Not in Stock"

def find_most_similar_user(target_user_features, user_features_dict):
    """Find the most similar user based on cosine similarity."""
    all_users = list(user_features_dict.keys())
    all_features = np.array([user_features_dict[uid] for uid in all_users])

    if all_features.shape[0] == 0:
        return None

    similarities = cosine_similarity(target_user_features.reshape(1, -1), all_features).flatten()
    best_match_idx = np.argmax(similarities)
    return all_users[best_match_idx]

def get_global_recommendations(topk=5):
    """Recommend the most popular products globally."""
    popular_products = (
        transactions_df.groupby("ProductID")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(topk)
        .index.tolist()
    )
    return popular_products

def get_recommendations(client_id, topk=5):
    """Recommend items for a user. Handles cold start cases."""
    if client_id in user2idx:
        u_idx = user2idx[client_id]
        user_tensor = torch.LongTensor([u_idx])
        all_item_indices = torch.arange(len(idx2item))

        hybrid_model.eval()
        with torch.no_grad():
            scores = hybrid_model(user_tensor.repeat(len(idx2item)), all_item_indices)

        topk_idx = torch.argsort(scores, descending=True)[:topk]
        return [idx2item[int(idx)] for idx in topk_idx]

    return get_global_recommendations(topk)

# ----------------------------------------------------
# 3. Streamlit App UI
# ----------------------------------------------------

st.markdown("### Enter your Client ID to get personalized recommendations!")

client_id_input = st.text_input("Client ID:", placeholder="Enter your numeric Client ID")

if st.button(" Get Recommendations"):
    if client_id_input.strip() == "":
        st.warning("⚠️ Please enter a valid Client ID.")
    else:
        try:
            client_id = int(client_id_input)
        except ValueError:
            st.error("⚠️ Client ID must be numeric.")
            st.stop()

        with st.spinner("🔍 Fetching recommendations..."):
            rec_items = get_recommendations(client_id, topk=5)

        if not rec_items:
            st.error("❌ No recommendations found for this Client ID.")
        else:
            st.success("🎯 Recommended Products:")
            for product_id in rec_items:
                product_name = get_family_level2(product_id)
                stock_status = check_stock(product_id)

                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="color: #333;">{product_name}</h4>
                    <p style="color: {'green' if stock_status == '✅ In Stock' else 'red'};">{stock_status}</p>
                </div>
                """, unsafe_allow_html=True)