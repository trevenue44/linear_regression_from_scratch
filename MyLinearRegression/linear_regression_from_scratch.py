import numpy as np


class CustomLinearRegression:
    """
    X should be an m by n matrix where m is the number of samples and n is the number of features.
    y should be an m by 1 matrix where m is the number of samples
    """

    def __init__(self, alpha=0.01, num_epochs=1000):
        # setting some initial values
        self._alpha = alpha
        self._num_epochs = num_epochs

    def _hypothesis(self):
        return np.dot(self._X, self._theta)

    def _cost(self):
        return (1 / (2 * self._m)) * np.sum(np.square(self._hypothesis() - self._y))

    def _get_new_theta(self):
        return self._theta - (self._alpha / self._m) * np.dot(
            self._X.T, self._hypothesis() - self._y
        )

    def _gradient_descent(self):
        for _ in range(self._num_epochs):
            self._theta = self._get_new_theta()

    def fit(self, X, y):
        # changing X to the design matrix
        self._X = np.insert(np.array(X), 0, 1, axis=1)

        self._y = np.array(y)

        # get the number of samples and features
        self._m = len(self._X)
        self._n = len(self._X[0])

        # initialize the weights and bias
        self._theta = np.array([[0] for _ in range(self._n)])

        self._gradient_descent()

    def predict(self, X):
        # changing X to the design matrix
        X = np.insert(np.array(X), 0, 1, axis=1)
        return np.dot(X, self._theta)

    def score(self, X, y):
        # getting the models predictions
        y_pred = self.predict(X)
        # calculating the error between the predictions and the actual values
        return 1 - (np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y))))

    def get_params(self):
        return self._theta
