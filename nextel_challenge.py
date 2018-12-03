import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn import ensemble
from sklearn.externals import joblib


df = pd.read_csv('house_sales.csv', sep=',', header=0).replace(np.NaN, 0) #read csv file

data = df.drop(['price'], axis=1) #remove these three columns
#data = data.div(data.sum(axis=1), axis=0) #normalization decreases the performance

dataNoLabels = np.asmatrix(data) #data: dataframe to nArray
prices = np.asmatrix(df['price']).transpose() #labels: dataframe to nArray
features = list(data)

X_train, X_test, y_train, y_test = train_test_split(dataNoLabels, prices, test_size=0.15, random_state=0) #split data to train and test

regr = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth= 5).fit(X_train,np.ravel(y_train,order='C')) #train the regressor
predictions = regr.predict(X_test) #test regressor
score = metrics.explained_variance_score(y_test,predictions)
print('Split Score: %.2f' % (score))

cv_results = cross_validate(regr, dataNoLabels, np.ravel(prices,order='C'), cv=5, return_train_score=False)
print('Cross Validation Score: %.2f' % (cv_results['test_score'].mean()))

joblib.dump(regr, 'nextelModel.pkl') #salva o modelo criado como um arquivo pkl

#COMPARE TRUE VS PREDICTED PRICES OF 100 TEST INSTANCES
plt.figure()
plt.plot(y_test[:100], 'r', label='True Prices')
plt.plot(predictions[:100], linestyle='dashed', label='Predicted Prices')
plt.legend()
plt.xlabel('Instances')
plt.ylabel('Prices')
plt.title('PRICES: True vs Predicted')
plt.savefig('true_predicted_prices.png')