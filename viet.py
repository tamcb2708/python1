import numpy as np
import pandas as pd


df_house = pd.read_csv("data/housing.csv")

#df_house = df_house.sample(frac=1)
print (df_house.shape)
print (df_house.head(10))
list = ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms",
"Area Population"]

X = df_house[list]
print (X.head(10))

y = df_house["Price"]
print (y.head(10))

from sklearn.decomposition import PCA
X = PCA(1).fit_transform(X)
print (X[:10])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=35)

from sklearn import linear_model
regr = linear_model.LinearRegression().fit(X_train, y_train)

y_pred = regr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Bias: \n', regr.intercept_)
# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color='red')
plt.scatter(X_train, regr.predict(X_train), color='green')
plt.scatter(X_test[:10,:], y_test[:10], color='black')
plt.title('hồi quy tuyến tính cho giá nhà')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


plt.plot([min(y_test), max(y_test)],[min(y_pred),max(y_pred)])
plt.scatter(y_test, y_pred, color='red')
plt.title('Compare')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()