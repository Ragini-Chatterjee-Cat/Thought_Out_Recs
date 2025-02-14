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
# 1. Define Hybrid Dataset Class (Uses Preferences)
# ----------------------------------------------------
class HybridDataset(Dataset):
    """PyTorch Dataset for hybrid recommendation model with preferences."""

    def __init__(self, pairwise_pdf, user2idx, item2idx, user_features_dict):
        self.user_ids = []
        self.pref_item_ids = []
        self.dispref_item_ids = []
        self.user_side_features = []

        for _, row in pairwise_pdf.iterrows():
            user_id = user2idx[row.ClientID]
            pref_item = item2idx[row.preferred_item]
            dispref_item = item2idx[row.dispreferred_item]

            self.user_ids.append(user_id)
            self.pref_item_ids.append(pref_item)
            self.dispref_item_ids.append(dispref_item)

            # Add metadata embeddings
            self.user_side_features.append(user_features_dict.get(row.ClientID, np.zeros(17)))

        self.user_ids = torch.LongTensor(self.user_ids)
        self.pref_item_ids = torch.LongTensor(self.pref_item_ids)
        self.dispref_item_ids = torch.LongTensor(self.dispref_item_ids)
        self.user_side_features = torch.tensor(np.array(self.user_side_features), dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.pref_item_ids[idx], self.dispref_item_ids[idx], self.user_side_features[idx]


# ----------------------------------------------------
# 2. Define Hybrid Model (CF + Metadata + Preferences)
# ----------------------------------------------------
class HybridPreferenceModel(nn.Module):
    """Hybrid Recommendation Model with collaborative filtering and metadata."""

    def __init__(self, n_users, n_items, user_feat_dim, embedding_dim=16):
        super().__init__()
        self.cf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.cf_item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.cf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.cf_item_embedding.weight, std=0.01)

        # Content-Based Filtering (CBF) - User Metadata Processing
        self.user_mlp = nn.Sequential(
            nn.Linear(user_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, user_ids, item_ids, user_features=None):
        cf_user = self.cf_user_embedding(user_ids)
        cf_item = self.cf_item_embedding(item_ids)

        # Content-based user embeddings
        content_user = self.user_mlp(user_features) if user_features is not None else 0

        # Final user representation (Fusion of CF & CBF)
        final_user = cf_user + content_user

        # Compute scores (dot product between user & item embeddings)
        scores = (final_user * cf_item).sum(dim=1)
        return scores


# ----------------------------------------------------
# 3. Define Hybrid Pairwise Loss (Preference-Based)
# ----------------------------------------------------
def hybrid_pairwise_kl_loss(model, reference, user_ids, pref_item, dispref_item, user_features, alpha=0.1):
    """Computes Pairwise + KL Loss using preferences."""
    s_pref = model(user_ids, pref_item, user_features)
    s_dispref = model(user_ids, dispref_item, user_features)
    diff = s_pref - s_dispref
    pairwise_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

    with torch.no_grad():
        s_pref_ref = reference(user_ids, pref_item, user_features)
        s_dispref_ref = reference(user_ids, dispref_item, user_features)

    kl_loss = ((s_pref_ref - s_pref) ** 2 + (s_dispref_ref - s_dispref) ** 2).mean()

    total_loss = pairwise_loss + alpha * kl_loss
    return total_loss, pairwise_loss.item(), kl_loss.item()


# ----------------------------------------------------
# 4. Data Preparation Function
# ----------------------------------------------------
def prepare_data():
    """Loads and preprocesses data for training."""
    print("🔹 Loading dataset...")
    data_frames = load_data()

    pairwise_pdf = data_frames["pairwise"]
    clients_pdf = data_frames["clients"]

    unique_users = pairwise_pdf["ClientID"].unique()
    unique_items = pd.unique(pd.concat([pairwise_pdf["preferred_item"], pairwise_pdf["dispreferred_item"]]))

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}

    # Create user metadata dictionary
    user_features_dict = {
        row.ClientID: np.concatenate([
            one_hot_encode(row.ClientSegment, segment2idx),
            one_hot_encode(row.ClientCountry, country2idx),
            one_hot_encode(row.ClientGender, gender2idx)
        ])
        for _, row in clients_pdf.iterrows()
    }

    dataset = HybridDataset(pairwise_pdf, user2idx, item2idx, user_features_dict)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader, len(user2idx), len(item2idx), user2idx, item2idx


# ----------------------------------------------------
# 5. Train the Model (Pairwise Preference Learning)
# ----------------------------------------------------
def train_model(model, train_loader, test_loader, epochs=10, alpha=0.05):
    """Trains the hybrid recommendation model using preference-based loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for user_ids, pref_item, dispref_item, user_features in train_loader:
            optimizer.zero_grad()
            loss, _, _ = hybrid_pairwise_kl_loss(model, model, user_ids, pref_item, dispref_item, user_features, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

    torch.save(model, MODEL_PATH)
    print(f" Model trained and saved at: {MODEL_PATH}")

    return train_losses


# ----------------------------------------------------
# 6. Main Execution
# ----------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, num_users, num_items, user2idx, item2idx = prepare_data()

    if os.path.exists(MODEL_PATH):
        print(f" Found existing Hybrid model: {MODEL_PATH}")
        hybrid_model = HybridPreferenceModel(num_users, num_items, 17)
        hybrid_model.load_state_dict(torch.load(MODEL_PATH).state_dict())
        hybrid_model.eval()
    else:
        print(" No trained model found. Training a new one...")
        hybrid_model = HybridPreferenceModel(num_users, num_items, 17)
        train_model(hybrid_model, train_loader, test_loader)
