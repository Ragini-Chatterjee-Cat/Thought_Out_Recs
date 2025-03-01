import os
import torch
from NCF import NeuralCollaborativeFiltering, prepare_data as prepare_ncf_data, train_model as train_ncf, \
    load_trained_model as load_ncf_model, recommend_for_user as recommend_ncf
from HybridModel import HybridPreferenceModel, prepare_data as prepare_hybrid_data, train_model as train_hybrid, \
    load_trained_model as load_hybrid_model, recommend_for_user as recommend_hybrid
from data_loader import load_data

# Define model paths
NCF_MODEL_PATH = "ncf_model.pth"
HYBRID_MODEL_PATH = "hybrid_model.pt"

if __name__ == "__main__":
    # Load Data
    print("🔹 Loading dataset...")
    data_frames = load_data()

    ### --- NCF Model Execution ---
    print("\n=== 🔹 Processing NCF Model ===")
    train_loader_ncf, test_loader_ncf, num_users_ncf, num_items_ncf, client_mapping_ncf, product_mapping_ncf = prepare_ncf_data(
        data_frames)

    if os.path.exists(NCF_MODEL_PATH):
        print(f"✅ Found existing NCF model: {NCF_MODEL_PATH}")

        # Load trained NCF model
        ncf_model = load_ncf_model(NCF_MODEL_PATH, num_users_ncf, num_items_ncf)

        # Test Recommendation
        test_user_id = 7673687066317773168  # Example User ID
        recommendations_ncf = recommend_ncf(ncf_model, client_mapping_ncf, product_mapping_ncf, test_user_id)

        print(f"📌 [NCF] Recommended Products for User {test_user_id}: {recommendations_ncf}")

    else:
        print("🚀 No trained NCF model found. Training a new model...")

        # Train new NCF model
        ncf_model = NeuralCollaborativeFiltering(num_users_ncf, num_items_ncf)
        train_ncf(ncf_model, train_loader_ncf, test_loader_ncf)

        print(f"✅ NCF Model trained and saved at: {NCF_MODEL_PATH}")

        # Load the newly trained model
        ncf_model = load_ncf_model(NCF_MODEL_PATH, num_users_ncf, num_items_ncf)

        # Test Recommendation
        recommendations_ncf = recommend_ncf(ncf_model, client_mapping_ncf, product_mapping_ncf, test_user_id)

        print(f"📌 [NCF] Recommended Products for User {test_user_id}: {recommendations_ncf}")

    ### --- Hybrid Model Execution ---
    print("\n=== 🔹 Processing Hybrid Model ===")
    train_loader_hybrid, test_loader_hybrid, num_users_hybrid, num_items_hybrid, client_mapping_hybrid, product_mapping_hybrid = prepare_hybrid_data()

    if os.path.exists(HYBRID_MODEL_PATH):
        print(f"✅ Found existing Hybrid model: {HYBRID_MODEL_PATH}")

        # Load trained Hybrid model
        hybrid_model = load_hybrid_model(HYBRID_MODEL_PATH, num_users_hybrid, num_items_hybrid)

        # Test Recommendation
        recommendations_hybrid = recommend_hybrid(hybrid_model, client_mapping_hybrid, product_mapping_hybrid,
                                                  test_user_id)

        print(f"📌 [Hybrid] Recommended Products for User {test_user_id}: {recommendations_hybrid}")

    else:
        print("🚀 No trained Hybrid model found. Training a new model...")

        # Train new Hybrid model
        hybrid_model = HybridPreferenceModel(num_users_hybrid, num_items_hybrid)
        train_hybrid(hybrid_model, train_loader_hybrid, test_loader_hybrid)

        print(f"✅ Hybrid Model trained and saved at: {HYBRID_MODEL_PATH}")

        # Load the newly trained model
        hybrid_model = load_hybrid_model(HYBRID_MODEL_PATH, num_users_hybrid, num_items_hybrid)

        # Test Recommendation
        recommendations_hybrid = recommend_hybrid(hybrid_model, client_mapping_hybrid, product_mapping_hybrid,
                                                  test_user_id)

        print(f"📌 [Hybrid] Recommended Products for User {test_user_id}: {recommendations_hybrid}")