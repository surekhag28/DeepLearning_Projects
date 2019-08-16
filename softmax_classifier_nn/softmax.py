
import numpy as np
from random import shuffle
import pdb

def softmax_loss(W, b, X, y, reg):
  """
  Softmax loss function.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - b: A numpy array of shape (C,) containing biases. 
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  - gradient with respect to weights b; an array of same shape as b

  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  db = np.zeros_like(b)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no for loops.       #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  scores = X.dot(W)
  exp_scores = np.exp(scores - np.max(scores))
  
  norm_exp_scores = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
  neg_log_likelihood = - np.log(norm_exp_scores[range(num_train),y])
  
  row_index = np.arange(num_train)
  
  
  data_loss = np.sum(neg_log_likelihood) / num_train
  loss = data_loss  + 0.5 * reg * np.sum(W * W)
  norm_exp_scores[row_index, y] -= 1

  dW = X.T.dot(norm_exp_scores)
  dW = dW/num_train + reg * W
  
  db = np.sum(norm_exp_scores, axis=1, keepdims=True)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  
    
  return loss, dW, db


