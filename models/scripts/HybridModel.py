import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from data_loader import load_data  # Import preprocessed data

# Define model path
MODEL_PATH = "hybrid_model.pth"


# ----------------------------------------------------
# 1. Define Hybrid Dataset Class
# ----------------------------------------------------
class HybridDataset(Dataset):
    """PyTorch Dataset for hybrid recommendation model."""

    def __init__(self, transactions_df, user2idx, item2idx):
        self.user_ids = torch.LongTensor([user2idx[row.ClientID] for _, row in transactions_df.iterrows()])
        self.item_i_ids = torch.LongTensor([item2idx[row.ProductID] for _, row in transactions_df.iterrows()])
        self.item_j_ids = torch.LongTensor([item2idx[row.ProductID] for _, row in transactions_df.iterrows()])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_i_ids[idx], self.item_j_ids[idx]


# ----------------------------------------------------
# 2. Define Hybrid Model
# ----------------------------------------------------
class HybridPreferenceModel(nn.Module):
    """Hybrid Recommendation Model with collaborative embeddings."""

    def __init__(self, n_users, n_items, embedding_dim=16):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        scores = (user_emb * item_emb).sum(dim=1)  # Dot product similarity
        return scores


# ----------------------------------------------------
# 3. Define Loss Function
# ----------------------------------------------------
def hybrid_pairwise_kl_loss(model, reference, user_ids, item_i, item_j, alpha=0.1):
    """Computes Pairwise + KL Loss."""
    s_i = model(user_ids, item_i)
    s_j = model(user_ids, item_j)
    diff = s_i - s_j
    pairwise_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

    with torch.no_grad():
        s_i_ref = reference(user_ids, item_i)
        s_j_ref = reference(user_ids, item_j)

    kl_loss = ((s_i_ref - s_i) ** 2 + (s_j_ref - s_j) ** 2).mean()

    total_loss = pairwise_loss + alpha * kl_loss
    return total_loss, pairwise_loss.item(), kl_loss.item()


# ----------------------------------------------------
# 4. Data Preparation Function
# ----------------------------------------------------
def prepare_data():
    """Loads and preprocesses data for training."""
    print("🔹 Loading dataset...")
    data_frames = load_data()

    transactions_pdf = data_frames["transactions"]
    unique_users = transactions_pdf["ClientID"].unique()
    unique_items = transactions_pdf["ProductID"].unique()

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}

    dataset = HybridDataset(transactions_pdf, user2idx, item2idx)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader, len(user2idx), len(item2idx), user2idx, item2idx


# ----------------------------------------------------
# 5. Model Training Function
# ----------------------------------------------------
def train_model(model, train_loader, test_loader, epochs=10, alpha=0.05):
    """Trains the hybrid recommendation model."""
    print("🚀 Training Hybrid Model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for user_ids, item_i, item_j in train_loader:
            optimizer.zero_grad()
            loss, _, _ = hybrid_pairwise_kl_loss(model, model, user_ids, item_i, item_j, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

    # Save Model
    torch.save(model, MODEL_PATH)
    print(f"✅ Model trained and saved at: {MODEL_PATH}")

    return train_losses


# ----------------------------------------------------
# 6. Load Trained Model Function
# ----------------------------------------------------
def load_trained_model(model_path, num_users, num_items):
    """Loads a trained hybrid recommendation model."""
    print(f"📂 Loading model from {model_path}...")
    model = HybridPreferenceModel(num_users, num_items)
    model.load_state_dict(torch.load(model_path).state_dict())
    model.eval()
    print("✅ Model loaded successfully!")
    return model


# ----------------------------------------------------
# 7. Recommendation Function
# ----------------------------------------------------
def recommend_for_user(model, user2idx, item2idx, user_id, top_n=5):
    """Recommends top N products for a given user."""
    if user_id not in user2idx:
        print("⚠️ User ID not found!")
        return None

    user_idx = user2idx[user_id]
    all_items = torch.arange(len(item2idx))

    # Predict scores for all items
    user_tensor = torch.LongTensor([user_idx]).repeat(len(item2idx))
    with torch.no_grad():
        scores = model(user_tensor, all_items)

    # Rank items by score
    top_item_indices = torch.argsort(scores, descending=True)[:top_n]
    recommended_product_ids = [iid for iid, idx in item2idx.items() if idx in top_item_indices]

    return recommended_product_ids


# ----------------------------------------------------
# 8. Main Execution
# ----------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, num_users, num_items, user2idx, item2idx = prepare_data()

    if os.path.exists(MODEL_PATH):
        print(f"✅ Found existing Hybrid model: {MODEL_PATH}")
        hybrid_model = load_trained_model(MODEL_PATH, num_users, num_items)
    else:
        print("🚀 No trained model found. Training a new one...")
        hybrid_model = HybridPreferenceModel(num_users, num_items)
        train_model(hybrid_model, train_loader, test_loader)

    # Test recommendation
    test_user_id = 7673687066317773168  # Example user
    recommendations = recommend_for_user(hybrid_model, user2idx, item2idx, test_user_id)
    print(f"📌 Recommended Products for User {test_user_id}: {recommendations}")