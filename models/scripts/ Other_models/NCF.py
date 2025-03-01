import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from data_loader import load_data  # Use the modularized data loader


class NCFDataset(Dataset):
    """ PyTorch Dataset for Neural Collaborative Filtering """

    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        """
        Initialize NCF Model.

        Args:
            num_users (int): Total number of users.
            num_items (int): Total number of items.
            embedding_dim (int): Size of embedding vectors.
        """
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP Layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        """
        Forward pass of the model.

        Args:
            user_ids (Tensor): Tensor of user IDs.
            item_ids (Tensor): Tensor of item IDs.

        Returns:
            Tensor: Predicted scores.
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(x).squeeze()


def prepare_data(data_frames, num_neg_samples=4):
    """
    Prepare training data with positive & negative samples.
    Returns:
        train_loader, test_loader, num_users, num_items
    """
    transactions_ncf = data_frames['transactions']

    # Step 1: Positive samples (purchased items)
    interactions = transactions_ncf[['ClientID', 'ProductID']].drop_duplicates()
    interactions['Label'] = 1  # Purchase happened

    # Step 2: Negative samples (unseen items)
    all_product_ids = transactions_ncf['ProductID'].unique()
    negative_samples = []

    for client in interactions['ClientID'].unique():
        purchased = set(interactions[interactions['ClientID'] == client]['ProductID'])
        for _ in range(num_neg_samples):
            neg_product = np.random.choice(all_product_ids)
            while neg_product in purchased:
                neg_product = np.random.choice(all_product_ids)
            negative_samples.append([client, neg_product, 0])

    negative_samples_df = pd.DataFrame(negative_samples, columns=['ClientID', 'ProductID', 'Label'])

    # Combine positive & negative samples
    dataset = pd.concat([interactions, negative_samples_df])

    # Encode ClientID & ProductID
    client_mapping = {id: idx for idx, id in enumerate(dataset['ClientID'].unique())}
    product_mapping = {id: idx for idx, id in enumerate(dataset['ProductID'].unique())}

    dataset['ClientID'] = dataset['ClientID'].map(client_mapping)
    dataset['ProductID'] = dataset['ProductID'].map(product_mapping)

    # Split data
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create PyTorch Datasets
    train_dataset = NCFDataset(train['ClientID'].values, train['ProductID'].values, train['Label'].values)
    test_dataset = NCFDataset(test['ClientID'].values, test['ProductID'].values, test['Label'].values)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    num_users = len(client_mapping)
    num_items = len(product_mapping)

    return train_loader, test_loader, num_users, num_items, client_mapping, product_mapping


def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    """
    Train the NCF Model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_ids, item_ids, labels in train_loader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "ncf_model.pth")
    print("Model training complete & saved!")


def load_trained_model(model_path, num_users, num_items, embedding_dim=32):
    """
    Load a trained NCF model.
    """
    model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def recommend_for_user(model, client_mapping, product_mapping, user_id, top_n=5):
    """
    Recommend top-N items for a given user.
    """
    if user_id not in client_mapping:
        print("User ID not found!")
        return None

    user_idx = torch.LongTensor([client_mapping[user_id]])
    all_products = torch.LongTensor(list(product_mapping.values()))

    with torch.no_grad():
        scores = model(user_idx.repeat(len(all_products)), all_products)

    top_products = all_products[torch.argsort(scores, descending=True)[:top_n]]
    recommended_product_ids = [pid for pid, idx in product_mapping.items() if idx in top_products.tolist()]

    return recommended_product_ids


# --- Main Execution ---
if __name__ == "__main__":
    # Load Data
    data_frames = load_data()

    # Prepare Data
    train_loader, test_loader, num_users, num_items, client_mapping, product_mapping = prepare_data(data_frames)

    # Initialize & Train Model
    ncf_model = NeuralCollaborativeFiltering(num_users, num_items)
    train_model(ncf_model, train_loader, test_loader)

    # Load Trained Model
    trained_model = load_trained_model("ncf_model.pth", num_users, num_items)

    # Test Recommendation
    test_user_id = 7673687066317773168
    recommendations = recommend_for_user(trained_model, client_mapping, product_mapping, test_user_id)
    print("Recommended Products:", recommendations)