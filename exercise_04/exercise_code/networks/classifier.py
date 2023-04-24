
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super(Classifier, self).__init__("classifier")

        self.num_features = num_features
        self.W = None

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point.
        :return: Predicted labels for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        
        # output variable
        y = None

        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the model. Note  #
        # that you need to implement the function self.sigmoid() for that.     #
        # Also, save in self.cache an array of all the relevant variables that #
        # you will need to use in the backward() function. E.g.: (X, ...)      #
        ########################################################################

        # X * W
        xw = X.dot(self.W)

        # save the xw matrix to cache for later use
        

        # apply sigmoid
        y = self.sigmoid(xw)
        self.cache =(X, xw) 
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return y

    def backward(self, dout):
        """
        Performs the backward pass of the model.

        :param dout: N x M array. Upsteam derivative. If the output of forward() is z, then it is dL/dz, where L is the loss function.
        :return: dW --> Gradient of the weight matrix, w.r.t the upstream gradient 'dout'.
        """
        assert self.cache is not None, "Run a forward pass before the backward pass. Also, don't forget to store the relevat variables\
            such as in 'self.cache = (X, y, ...)"
        dW = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient wrt W, dW           #
        # Make sure you've stored ALL needed variables in self.cache.          #
        # Hint 1: It is recommended to follow the Stanford article on          #
        # calculating the chain-rule, while dealing with matrix notations:     #
        # http://cs231n.stanford.edu/handouts/linear-backprop.pdf              #
        #                                                                      #   
        # Hint 2: Remember that the derivative of sigmoid(x) is independent of #
        # x, and could be calculated with the result, calculated earlier at    #
        # the forward() function.                                              #
        ########################################################################

        # L(w) = L(sigmod(xw), Y)
        # dL(w) /dw = dL(w)/d(y') * d(y')/dw
        #
        # derivative of sigmoid(s) w.r.t s and derivative of X*w w.r.t w

        # y = xw, z = sigmoid(y)

        print("below we extract the cache")
        x, y = self.cache
        sig_y = self.sigmoid(y)

        # derivative of z w.r.t y
        dz_dy = sig_y * (1 - sig_y)

        # derivative of L w.r.t y
        dL_dy = dout * dz_dy

        # standford trick
        dW = x.T.dot(dL_dy)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dW

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None

        ########################################################################
        # TODO:                                                                #
        # Implement the sigmoid function over the input x.                     #
        # Return out.                                                          #
        ########################################################################


        out = 1 / (1 + np.e ** -x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))
