# Thought Out Recs - Hybrid Recommendation System

A sophisticated hybrid recommendation system combining collaborative filtering and content-based filtering techniques. Built with PyTorch and deployed as an interactive Streamlit web application.

**Note**: This project was created as an academic project. The original dataset is not included in this repository. To use this system, you'll need to provide your own data (see [Data Requirements](#data-format) below).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run models/scripts/RC-1.py
```
## The Data
We used **Spark** to load the data as it was the only way we could effectively handle large datasets. We then merged the data using the **IDs**, focusing on **transactions** to ensure our model learns from customer behavior. 

Our goal is to capture **customer preferences**, as raw purchase quantity alone does not indicate true preference. For instance, a customer may frequently buy a **cheap phone charger** but might actually prefer a **premium brand** they can’t afford as often. We aim to **recommend the premium brand** based on behavioral patterns.

We used **quantity as a metric** to gauge preference. We aggregated the **total quantity** bought by a client for each product and retained only the **top 50 items per user** to reduce noise. 

To generate training data, we created **preference pairs**: if **Item A is bought more than Item B**, we create a **(preferred, dispreferred) pair (A, B)** where **quantity(A) > quantity(B)**.

To train the neural network, we **converted these items into numerical indices**. Each user and item received a **unique integer index**, which was used for **embedding lookups** in the model.

---

## Including Metadata for Cold-Start Support
To address the **cold-start problem**, we included user attributes (**segment, country, gender**) and applied **one-hot encoding** to create feature vectors. We excluded **age** as it was not usable. 

Users are now represented **not only by their ID but also by their metadata embeddings**. However, users must have **valid metadata** for this approach to work effectively.Note:  If compute_user_features() returns all zeros, the content-based branch doesn't contribute to personalization.

After processing, we **formatted our data as tensors** and split it into **80% training and 20% testing**.

---

## The Model
We tried multiple models like: 
1. Item-based Collaborative Filtering + Association Rule (predictions can be found in the zipped json files)
2.  ALS (Alternating Least Squares Algorithm)
3.  Graph Neural Network + Cold Start
4.  NCF (files can also be found in the repo)
5.  The Hybrid Recommendation: the following sections talk about that:

   
Our model is a **Hybrid Recommendation System** that combines:
1. **Collaborative Filtering (CF)** for **user-item interactions**.
2. **Content-Based Filtering (CBF)** to **process user metadata**.
3. A **final ranking model** that fuses both CF and CBF representations.

- Users have **two representations** (one from CF, one from metadata).
- Items have **one learned representation**.
- The **dot product** between user and item embeddings determines preference scores.

---

## Loss Functions
We use **two loss functions**:

### **1. Pairwise Ranking Loss**
Ensures the **preferred item** has a **higher score** than the **dispreferred item**. 
- The model assigns scores to both items and maximizes the probability that the **preferred item** ranks higher.

### **2. KL-Divergence Loss (Regularization)**
- Prevents **overfitting** by aligning the model’s predictions with a reference model.
- Helps stabilize learning and avoid extreme predictions.
- Both losses are combined with **alpha = 0.1**.

We trained the model for **10 epochs**, as validation loss increased beyond this point.

---

## Project Structure

```
Thought_Out_Recs/
├── Hybrid_model.ipynb        # Main training notebook
├── models/
│   └── scripts/
│       ├── RC-1.py           # Streamlit application
│       ├── Other_models/     # Alternative model implementations
│       ├── hybrid_model.pth  # Trained model weights
│       └── mappings.pkl      # User/item ID mappings
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Apache Spark 3.3+ (for data processing)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Thought_Out_Recs.git
cd Thought_Out_Recs
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Prepare data**:
   - Place your CSV files (clients.csv, products.csv, stocks.csv, stores.csv, transactions.csv) in your desired directory
   - Update the `SHARED_FOLDER_PATH` variable in `models/scripts/data_loader.py` to point to your data directory

## Usage

### Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook Hybrid_model.ipynb
```

The notebook includes:
1. Data loading with Spark
2. Preference pair generation
3. Model training
4. Evaluation
5. Model saving

### Running the Streamlit App

```bash
streamlit run models/scripts/RC-1.py
```

The web interface allows you to:
- Enter a Client ID
- Get personalized product recommendations
- View stock availability
- Explore recommendation scores

---

## Streamlit Integration
We built a **Streamlit web app** for real-time recommendations:
1. **Loaded the trained hybrid recommendation model and dataset mappings**.
2. **Created an interactive UI** where users can input their **Client ID** to receive **personalized recommendations** based on past purchases and metadata.
3. **For existing users**, the trained model ranks items.
4. **For new users**, we recommend the **most popular products globally** as we didn't have their metadata.
5. Implemented a **stock availability check** and **displayed recommendations in a structured card format** with a **custom background and styled UI elements**.

## Features

- **Hybrid Architecture**: Combines collaborative and content-based filtering
- **Cold-Start Support**: Uses user metadata (segment, country, gender) for new users
- **Scalable Data Processing**: Leverages PySpark for large datasets
- **Interactive Web App**: Real-time recommendations via Streamlit
- **Stock Integration**: Checks product availability
- **Fallback Recommendations**: Popular products for users without metadata

## Model Performance

The hybrid model outperformed several baseline approaches:

1. **Item-based Collaborative Filtering + Association Rules**
2. **ALS (Alternating Least Squares)**
3. **Graph Neural Network + Cold Start**
4. **NCF (Neural Collaborative Filtering)**
5. **Hybrid Recommendation** (current implementation)

## Technical Details

### Model Architecture

```python
HybridRecommendationModel(
    num_users=NUM_USERS,
    num_items=NUM_ITEMS,
    user_features_dim=USER_METADATA_DIM,
    embedding_dim=64,
    hidden_dim=128
)
```

### Loss Functions

- **Pairwise Ranking Loss**: Ensures preferred items rank higher
- **KL-Divergence Loss**: Regularization to prevent overfitting
- Combined with alpha = 0.1

### Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Epochs**: 10 (with early stopping)
- **Batch Size**: Configurable
- **Train/Test Split**: 80/20

## Data Format

### Expected Input

**Transaction Data**:
- User ID
- Item ID
- Quantity purchased
- Timestamp (optional)

**User Metadata**:
- Segment
- Country
- Gender

**Item Metadata**:
- Product descriptions
- Categories
- Stock availability

## Dependencies

- **torch**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **streamlit**: Web application framework
- **pyspark**: Distributed data processing
- **matplotlib**: Visualization

See `requirements.txt` for specific versions.

## Alternative Models

The `Other_models/` directory contains implementations of:

- **NCF**: Neural Collaborative Filtering
- **HybridModel**: Alternative hybrid architectures
- **data_loader**: Data preprocessing utilities
- **evaluation**: Model evaluation metrics

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## License

MIT License - Educational and research purposes

## Acknowledgments

- Built with PyTorch and Streamlit
- Inspired by modern recommendation system research
- Uses collaborative and content-based filtering techniques

## Author

Ragini Chatterjee

## Citation

If you use this code in your research, please cite:

```bibtex
@software{thought_out_recs,
  author = {Chatterjee, Ragini},
  title = {Thought Out Recs: Hybrid Recommendation System},
  year = {2025},
  url = {https://github.com/yourusername/Thought_Out_Recs}
}
```

---

**Note**: This is a research/educational project demonstrating hybrid recommendation system techniques. Ensure you have appropriate data permissions before use.

