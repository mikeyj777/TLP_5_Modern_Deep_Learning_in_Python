import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import os


from util import getData, softmax, cost2, y2indicator, error_rate, relu

from sklearn.utils import shuffle


class ANN(object):

    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, learning_rate=1e-6, reg=1e-6,  epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)

        Nvalid = X.shape[0] // 5

        Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:]
        #Tvalid = y2indicator(Yvalid)
        X, Y = X[:-Nvalid], Y[:-Nvalid]

        N, D = X.shape

        K = len(set(Y))

        T = y2indicator(Y)

        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M + K)
        self.b2 = np.zeros(K)

        costs = []
        best_valid_err = 1

        t0 = datetime.now()

        for i in range(epochs):
            pY, Z = self.forward(X)

            # grad desc

            pY_T = pY - T

            self.W2 -= learning_rate * (Z.T.dot(pY_T) + reg * self.W2)
            self.b2 -= learning_rate * (pY_T.sum(axis=0) + reg * self.b2)

            #dZ = pY_T.dot(self.W2.T) * (Z > 0)
            dZ = pY_T.dot(self.W2.T) * (1 - Z * Z)  # tanh

            self.W1 -= learning_rate * (X.T.dot(dZ) + reg * self.W1)
            self.b1 -= learning_rate * ((dZ).sum(axis=0) + reg * self.b1)

            if i % 20 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))

                dt = datetime.now() - t0
                print("i:  ", i, ".  cost:  ", c,
                      ".  error:  ", e, ".  dt:  ", dt)
                t0 = datetime.now()
                if e < best_valid_err:
                    best_valid_err = e

        print("best valid err:  ", best_valid_err)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        # Z = relu(X.dot(self.W1)  + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():

    t0 = datetime.now()

    print("now:  ", t0)

    X, Y = getData()

    dt = datetime.now() - t0

    print("dt:  ", dt)

    M = 10  # hidden units

    model = ANN(M)
    model.fit(X, Y, show_fig=True)
    print(model.score(X, Y))


if __name__ == '__main__':
    main()
