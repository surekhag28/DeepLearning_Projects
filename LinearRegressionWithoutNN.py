import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score



def read_dataset():
    dataset = pd.read_csv("50_Startups.csv")
    X = dataset.iloc[:, :-1].values
    z = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    labelencoder = LabelEncoder()
    X[:,3] = labelencoder.fit_transform(X[:,3])
    onehotencoder = OneHotEncoder(categorical_features=[3])
    X = onehotencoder.fit_transform(X).toarray()
    return X,y


def fit_model(features,targets):
    regressor = LinearRegression()
    regressor.fit(features,targets)
    return regressor




X,y = read_dataset()
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=0)
model = fit_model(train_x,train_y)
y_pred = model.predict(test_x)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
print('Variance score: %.2f' % r2_score(test_y, y_pred))

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.scatter(test_y, y_pred, edgecolors=(0, 0, 0))
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax1.set_xlabel('Measured')
ax1.set_ylabel('Predicted')

ax2 = fig.add_subplot(222)
ax2.scatter(train_x[:,3],train_y,edgecolors=(0,0,0))
ax2.set_xlabel('R & D Spend')
ax2.set_ylabel('Profit')

ax3 = fig.add_subplot(223)
ax3.scatter(train_x[:,4],train_y,edgecolors=(0,0,0))
ax3.set_xlabel('Administration Spend')
ax3.set_ylabel('Profit')

ax4 = fig.add_subplot(224)
ax4.scatter(train_x[:,5],train_y,edgecolors=(0,0,0))
ax4.set_xlabel('Marketing Spend')
ax4.set_ylabel('Profit')

plt.tight_layout()
plt.show()


# Answer
# If we observer the plot "R & D Spend vs Profit", We can see the positive relation between them compared to "Administration Spend & Marketing Spend vs Profit".
# Hence company which spends more on R & D  performs better.








