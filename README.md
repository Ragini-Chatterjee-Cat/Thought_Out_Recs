# Thought Out Recs

## Run: streamlit run RC-1.py after cloning repository for the streamlit app
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

## Streamlit Integration
We built a **Streamlit web app** for real-time recommendations:
1. **Loaded the trained hybrid recommendation model and dataset mappings**.
2. **Created an interactive UI** where users can input their **Client ID** to receive **personalized recommendations** based on past purchases and metadata.
3. **For existing users**, the trained model ranks items.
4. **For new users**, we recommend the **most popular products globally** as we didn't have their metadata.
5. Implemented a **stock availability check** and **displayed recommendations in a structured card format** with a **custom background and styled UI elements**.

