import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import pandas as pd

from util import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator, initwb


def main():
    Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()

    print('logistic regression')

    # randomly assign weights

    N, D = Xtrain.shape

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    M = 10
    scale = 28

    # full grad descent
    W, b = initwb(D, M, scale)

    LL = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    for i in range(200):
        P_Y = forward(Xtrain, W, b)

        W += lr * (gradW(Ytrain_ind, P_Y, Xtrain) - reg * W)
        b += lr * (gradb(Ytrain_ind, P_Y) - reg * b)

        P_Y_test = forward(Xtest, W, b)
        ll = cost(P_Y_test, Ytest_ind)
        LL.append(ll)
        if i % 10 == 0:
            err = error_rate(P_Y_test, Ytest)
            print("cost at iter:  %d:  %.6f" % (i, ll))
            print("error rate:  ", err, "\n")

    P_Y = forward(Xtest, W, b)
    print("final error:  ", error_rate(P_Y, Ytest))
    print("elapsed time for full GD:  ", datetime.now() - t0)

    # 2.  Stochastic
    W, b = initwb(D, M, scale)

    LL_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in range(1):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for n in range(min(N, 500)):
            x = tmpX[n, :].reshape(1, D)
            y = tmpY[n, :].reshape(1, 10)
            P_Y = forward(x, W, b)

            W += lr * (gradW(y, P_Y, x) - reg * W)
            b += lr * (gradb(y, P_Y) - reg * b)

            P_Y_test = forward(Xtest, W, b)

            ll = cost(P_Y_test, Ytest_ind)

            LL_stochastic.append(ll)

            if n % (N / 2) == 0:
                err = error_rate(P_Y_test, Ytest)
                print("Cost at iteration %d:  %6.f" % (i, ll))
                print("error rate:  ", err)

    P_Y = forward(Xtest, W, b)

    print("error rate:  ", error_rate(P_Y, Ytest))
    print("elapsed time for SGD:  ", datetime.now() - t0)

    # batch
    W, b = initwb(D, M, scale)

    LL_batch = []
    lr = 0.001
    reg = 0.01
    batch_sz = 500
    n_batches = N // batch_sz

    t0 = datetime.now()
    for i in range(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j * batch_sz:(j * batch_sz + batch_sz), :]
            y = tmpY[j * batch_sz:(j * batch_sz + batch_sz), :]
            P_Y = forward(x, W, b)

            W += lr * (gradW(y, P_Y, x) - reg * W)
            b += lr * (gradb(y, P_Y) - reg * b)
            P_Y_test = forward(Xtest, W, b)

            ll = cost(P_Y_test, Ytest_ind)
            LL_batch.append(ll)
            if j % (n_batches / 2) == 0:
                err = error_rate(P_Y_test, Ytest)
                print("Cost at iteration %d:  %6.f" % (i, ll))
                print("error rate:  ", err)
    P_Y = forward(Xtest, W, b)

    print("error rate:  ", error_rate(P_Y, Ytest))
    print("elapsed time for SGD:  ", datetime.now() - t0)

    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
