
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# reading dataset
def read_boston_data():
    boston = load_boston()
    features = boston.data
    labels = boston.target
    return features,labels

# normalizing the features
def feature_normalize(features):
    mu = np.mean(features,axis=0)
    sigma = np.std(features,axis=0)
    features = (features - mu)/sigma
    return features

# creating biases to add to equation of line y = WX + b
def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l

X,y = read_boston_data()
X = feature_normalize(X)
X,y = append_bias_reshape(X,y)
n_dim = X.shape[1]

X_train,X_test,y_train,y_test = train_test_split( X, y , test_size=0.2, random_state=0)

learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,1]))


init = tf.global_variables_initializer()

y = tf.placeholder(tf.float32)
y_ = tf.matmul(X,W)
cost = tf.reduce_mean(tf.square(y_ - y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={X:X_train, y: y_train})
    cost_history = np.append(cost_history,sess.run(cost, feed_dict={X:X_train, y:y_train}))

plt.plot(range(len(cost_history)), cost_history)
plt.axis([0,training_epochs, 0, np.max(cost_history)])
plt.show()

# test model on test data set and calculate mean squared error

pred_y = sess.run(y_, feed_dict={X:X_test})
mse = tf.reduce_mean(tf.square(pred_y - y_test))
print("MSE: %.4f " % sess.run(mse))

fig, ax = plt.subplots()
ax.scatter(y_test,pred_y)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# prediction on sample data
x_sample = [[1. ,-0.41468015 ,-0.48772236, -1.30687771 ,-0.27259857 ,-0.83528384,
  1.01630251, -0.80988851 , 1.07773662 ,-0.75292215, -1.10611514 , 0.1130321,
  0.41616284, -1.36151682]]

y_pred_sample = tf.matmul(x_sample , sess.run(W))
print("Predicted price value %.2f" %sess.run(y_pred_sample))

sess.close()






