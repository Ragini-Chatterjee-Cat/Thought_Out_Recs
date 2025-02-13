# Thought_Out_Recs
The data: 
We used spark to load the data as it was the only way we could effectively deal with it. 
Then we merged the data using the IDs. We wanted to focus on transactions as we wanted our model to learn on customer behaviour. In this process we aim to learn customer preference because someone could buy a cheap phone charger every month (high quantity), but they may actually prefer a premium brand that they can’t afford to buy often and we would like to recoomend the premium brand. 

We used quanity as a metric to gauge preference. We get the total quantity bought by a client for each product and only keep the top 50 to reduce noise. Then we create preference pairs like f Item A is bought more than Item B, we create a (preferred, dispreferred) pair: (A,B). Where quantity of A is more that quantity of B. 

To train the neural network convert these to numerical indices.Each user and item has a unique integer index.These indices are used for embedding lookups in the model. 

INCLUDING META DATA FOR COLD START SUPPORT 

We also process user attributes (e.g., segment, country, gender) to create one-hot encoded features for cold-start generalization. We did not use age as it was not usable. Users are not just represented by their ID but also by their metadata embeddings. Note that users have to have valid meta data for this to work. 

After this we formated our data for training by storing everything as tensors. We split the dataset into 80% training and 20% testing.

THE MODEL 
Our model uses Collaborative Filtering (CF) for user-item interactions.And uses Content-Based Filtering (CBF) to process user metadata.Then it combines both into a final ranking model. Users have two representations (one from CF, one from metadata).
Items have one learned representation. The dot product determines preference scores.

LOSSES 
We use pairwise ranking loss to make sure the preferred item scores higher than the dispreferred item. For each user, we take a preferred item and a dispreferred item.The model assigns scores to both items.We maximize the probability that the preferred item has a higher score.

We have also used KL-Divergence (Kullback-Leibler Divergence) is used as a regularizer to prevent the model from overfitting. Both losses are combined during training. (alpha = 0.05) 

Then we trained the model for 10 epochs (as after this validation loss increased) 

STREAMLIT: 
We loaded the trained hybrid recommendation model and dataset mappings. We created an interactive UI where users can enter their Client ID to receive personalized recommendations based on their past purchases and metadata. For existing users, we used the trained model to rank items, while for new users, we recommended the most popular products globally. We also implemented a stock availability check and displayed recommendations in a structured card format with a custom background and styled UI elements.


