# Desafio_Nextel

Goal: Build a model that best predicts housing prices from that region.  
Run: “nextel_challenge.py”  
Libraries: “numpy”, “pandas”, “matplotlib.pyplot”, “sklearn”.  
Input: “house_sales.csv” (same directory as nextel_challenge.py).  
Processing:  
- Column ‘price’ is set as labels, and the remaining columns is used as train/test data.  
- The splitting of data is randomized being 85% as train data and 15% as test data.  
- To build the prediction model, it used the Gradient Boosting Regressor with 100 members.  
- Score by splitting: 88%  
- Cross Validation (6-fold) is also used in order to compare scores obtaining a mean score of 86%.  
- For better visualization of result, there is a plot of 100 predicted prices compared to their true prices.  
Output: “nextelModel.pkl”, “true_predicted_prices.png” (see below).  
