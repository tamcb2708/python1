
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def get_home_data():
    """Get home data, from local csv."""
    if os.path.exists("data/housing1.csv"):
        print("-- lấy từ dataset: home_data")
        df = pd.read_csv("data/housing1.csv", index_col=0)

    return df
df = get_home_data()
df.head()
df_house = df.sample(frac=1)
print (df_house.shape)
print (df_house.head(10))

list1 = ["Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms"]

X = df_house[list1]
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

# Tạo mô hình học model_lr: huấn luyện tập X_train với kỹ thuật học máy linear regression.
from sklearn.metrics import mean_squared_error, r2_score
# The coefficients
print('tạo mô hình học model_lr: huấn luyện tập X_train với kỹ thuật học máy linear regression \n')
print('nghiệm thuật toán: \n', regr.coef_)
print('Bias: \n', regr.intercept_)
# Sai số bình phương trung bình
print('Sai số bình phương trung bình: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Sai số hoàn hảo: %.2f'
      % r2_score(y_test, y_pred))
#

def plotting_features_vs_target(features, x, y):
    # define number of subplot
    num_feature = len(features)
    f, axes = plt.subplots(1, num_feature, sharey=True)

    # plotting
    for i in range(0, num_feature):
        # hiển thị giá tiền với số lượng X với giá tiền y
        axes[i].scatter(x[features[i]], y, color='blue')
        # hiển thị tên cột X
        axes[i].set_title(features[i])

        # dự đoán kết quả với giá trị X_test và y_test
        # if y_test >0:
        #        axes[i].scatter(X_test[:10,:], y_test[:10], color='black')
        # else:
        #     print('không thể dự đoán được')
        # axes[i].scatter(X_test[:10,:], y_test[:10], color='black')

        # dự đoán kết quả với giá trị X và y
        
        axes[i].scatter(X_train, regr.predict(X_train), color='green')
    plt.show()

if __name__ == "__main__":
    df = get_home_data()

    # features selection
    features = list(["Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms"])
    print("các cột:", list(df.columns.values))
    print("chọn từ các cột:", features)
    y = df["Price"]
    X = df[features]

    # split data-set into training (70%) and testing set (30%)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # plotting features, target relationships
    plotting_features_vs_target(features, x_train, y_train)


    # training model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # evaluating model
    score_trained = linear.score(x_test, y_test)
    print ("Mô hình đã ghi điểm:", score_trained)

    # L1 regularization
    lasso_linear = linear_model.Lasso(alpha=1.0)
    lasso_linear.fit(x_train, y_train)

    # evaluating L1 regularized model
    score_lasso_trained = lasso_linear.score(x_test, y_test)
    print ("Mô hình Lasso đã ghi điểm:", score_lasso_trained)


    # L2 regularization
    ridge_linear = Ridge(alpha=1.0)
    ridge_linear.fit(x_train, y_train)

    # evaluating L2 regularized model
    score_ridge_trained = ridge_linear.score(x_test, y_test)
    print ("Mô hình Ridge đã ghi điểm:", score_ridge_trained)



    poly_model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                           ('linear', linear_model.LinearRegression(fit_intercept=False))])
    poly_model = poly_model.fit(x_train, y_train)
    score_poly_trained = poly_model.score(x_test, y_test)
    print ("Mô hình poly ghi điểm:", score_poly_trained)

    poly_model = Pipeline([('poly', PolynomialFeatures(interaction_only=True, degree=2)),
                           ('linear', linear_model.LinearRegression(fit_intercept=False))])
    poly_model = poly_model.fit(x_train, y_train)
    score_poly_trained = poly_model.score(x_test, y_test)
    print ("Đã ghi điểm poly mô hình (chỉ tương tác):", score_poly_trained)
