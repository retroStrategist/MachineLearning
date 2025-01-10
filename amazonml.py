import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Paths to the files
amazon_products_path = os.path.join("datasets", "amazon_products.csv")
amazon_categories_path = os.path.join("datasets", "amazon_categories.csv")

#Reads the file and stores it in DataFrames
products_df = pd.read_csv(amazon_products_path)
categories_df = pd.read_csv(amazon_categories_path)
# print a summary of the data in Amazon

categories_dict = dict(zip(categories_df['id'], categories_df['category_name']))
# For printing all pairs: categories_dict = categories_df.to_dict('tight')

#for id in categories_dict:
#   print(categories_dict.get(id))
    
# print(products_df.columns)

# Column names:
# 'asin', 'title', 'imgUrl', 'productURL', 'stars', 'reviews', 'price', 
# 'listPrice', 'category_id', 'isBestSeller', 'boughtInLastMonth'

# Target parameter will be price
y = products_df.price
# Based on category_id, boughtInLastMonth, stars, reviews
features = ["category_id", "boughtInLastMonth", "stars", "reviews"]
X = products_df[features]

#Splits the data into training and validation sets
train_X, val_X, train_y, val_y =  train_test_split(X, y, random_state=5)

#Creates DecisionTreeRegression model
amazon_model = DecisionTreeRegressor(random_state=5)

#Makes model fit values
amazon_model.fit(train_X, train_y)

#Predicition comparison
val_predictions = amazon_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

# Definition of function to get MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [500, 550, 600, 650, 700, 750, 1000, 1500]
# Loop to find the ideal tree size from candidate_max_leaf_node
store_mae = 9999999999999
best_size = -1
for max_leaf_nodes in candidate_max_leaf_nodes:
    new_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if new_mae < store_mae:
        store_mae = new_mae
        best_size = max_leaf_nodes
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = best_size

print(f"Best leaf node max is: {best_tree_size}")
print("Validation MAE when finding the ideal max_leaf_nodes: {:,.0f}".format(val_mae))
