#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendations Using Collaborative Filtering

# ## Motivation
# 
# #### Overall Objective of the project: 
# 
# - Predict users' movie preferences and suggest a list of movies that are likely to be watched. In other words, give personalized suggestions on movies to watch for each user.
# 
# 
# ### Recommendation systems :
# 
# 
# #### What is a Recommendation system?
# 
# - A recommendation system suggests a product or service to customers who are likely to consume or purchase it.
# 
# - Recommendation systems are utilized in e-commerce to predict the preference a user might give to an item.
#   - Netflix : Which movie to watch.
#   - Amazon : Which products are similar to the one purchased.
# 
# - Other examples of recommendation systems and applications:
# 
#   - Spotify recommends music and playlist
#   - Facebook recommends friends
#   - LinkedIn recommends jobs
# 
# 
# - The Netflix prize (competition 2009)
# 
#     -  Competition for the best collaborative filtering algorithm to predict user ratings for films.
# 
#    -  Data: Around 100M ratings from 500K  users on 18K movies.
# 
#    -  Winning team used: A Regularized matrix factorization approach. 
#    
#    
# #### Recomendation system
# 
# Data required for Recommendation system:
# 
# - User ratings data
# - Variables related to items or users (movies genre, duration of the movie... etc)
# 
# 
# 
# #### Recomendation systems : Collaborative filtering
# 
# - **Main idea**: Recommending items to users based on the preference of similar users.
#    - Based on data, we asumme that: A user who has agreed in past tends to also agree in future.
#    
#    
# - We only have ratings of user for items.
#      - Users are consumers.
#      - Items are the products or services offered.
# 
# 
# - **Approach** : Build an "utility" matrix that captures interactions between users and items.
#    - each row is a user
#    - each column and item
#    
#    
# #### Collaborative filtering : Challenges
# 
# 
# - Sparsity of utility matrix:
#    - Usually users only interact with a few items.
#       - Netflix users rate only a few songs.
#    
# - Objective?
#  - Given a utility matrix of $N$ users and $M$ items, fill the the missing entries to complete the utility matrix. 
#  
#  
# #### Problem formulation
# 
# - An Unsupervised Learning approach
#    - Only uses the user-item utility matrix.
# 
# - Goal : learn latent features related to users and items.
#    - **Matrix factorization algorithm** on the utility matrix to learn latent features related to typical users and typical items.
#    - Use reconstructions to fill in missing entries.
#    
#    
#  #### Problem formulation - Matrix Factorization
# 
# 
# $$\hat{Y} = Z W$$
# $$Y \approx Z W$$
# 
# - $Z$ - Transformed data. Users to latent features of items
# - $W$ - Weights.    Items to latent features of users.
# 
# Adapted loss function (MSE) because of sparcity of data.
# 
# 
# $$\sum_{(i, j) \in R}  (W_{j}^T  Z_{i} - Y_{i,j})^2$$
# 
# - Where $R$ is the only available ratings
