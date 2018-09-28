import numpy as np
import matplotlib.pyplot as plt

from util_old import getData, softmax, cost, y2indicator, error_rate

from sklearn.utils import shuffle


class LogisticModel():

    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=10e-8, reg=10e-12, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape

        K = len(set(Y))

        T = y2indicator(Y)

        self.W = np.random.randn(D, K) / np.sqrt(D + K)
        self.b = np.zeros(K)

        costs = []
        best_valid_err = 1

        for i in range(epochs):
            pY = self.forward(X)

            # grad desc
            self.W -= learning_rate * (X.T.dot(pY - T) + reg * self.W)
            self.b -= learning_rate * ((pY - T).sum(axis=0) + reg * self.b)

            if i % 1 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost(Tvalid, pYvalid)
                costs.append(c)

                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                if e < 0.09:
                    break
                print("i:  ", i, ".  cost:  ", c, ".  error:  ", e)
                if e < best_valid_err:
                    best_valid_err = e

        print("best valid err:  ", best_valid_err)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)

#
# def main():
#     X, Y = getData()
#
#     model = LogisticModel()
#     model.fit(X, Y, show_fig=True)
#     print(model.score(X, Y))
#
#
# if __name__ == "__main__":
#     main()
