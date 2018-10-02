import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import pandas as pd

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_b1, derivative_b2, derivative_w1, derivative_w2


def main():

    doconstlearning = False
    dormsprop = True

    max_iter = 20  # make it 30 for sigmoid
    print_period = 50

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    lr = 0.00004
    reg = 0.01

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300
    K = 10
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    # save initial weights
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # batch

    #cost = -16
    if doconstlearning:
        print('const learning')
        losses_batch = []
        errors_batch = []

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
                Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
                pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

                W2 -= lr * (derivative_w2(Z, Ybatch, pYbatch) + reg * W2)
                b2 -= lr * (derivative_b2(Ybatch, pYbatch) + reg * b2)
                W1 -= lr * (derivative_w1(Xbatch, Z,
                                          Ybatch, pYbatch, W2) + reg * W1)
                b1 -= lr * (derivative_b1(Z,  Ybatch, pYbatch, W2) + reg * b1)

                if j % print_period == 0:
                    pY, _ = forward(Xtest, W1, b1, W2, b2)
                    l = cost(pY, Ytest_ind)
                    losses_batch.append(l)
                    print("cost at iter i %d, j %d:  %.6f" % (i, j, l))

                    e = error_rate(pY, Ytest)
                    errors_batch.append(e)

        pY, _ = forward(Xtest, W1, b1, W2, b2)
        print("final error rate:  ", error_rate(pY, Ytest))
        print()
        plt.plot(losses_batch, label='const')

    if dormsprop:
        print('rms prop')
        losses_rms = []
        errors_rms = []

        lr0 = 0.001
        cache_W2 = 1
        cache_b2 = 1
        cache_W1 = 1
        cache_b1 = 1
        decay_rate = 0.999
        eps = 1e-10

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
                Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
                pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

                gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
                cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
                W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

                gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
                cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
                b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)

                gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
                cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
                W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

                gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1
                cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
                b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)

                if j % print_period == 0:
                    # calculate just for LL
                    pY, _ = forward(Xtest, W1, b1, W2, b2)
                    # print "pY:", pY
                    ll = cost(pY, Ytest_ind)
                    losses_rms.append(ll)
                    print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll))

                    err = error_rate(pY, Ytest)
                    errors_rms.append(err)
                    print("Error rate:", err)

        plt.plot(losses_rms, label='rms')

        pY, _ = forward(Xtest, W1, b1, W2, b2)
        print("Final error rate:", error_rate(pY, Ytest))
        print()
        print("W1.shape", W1.shape)
        print("W1:  ", W1)
        print()
        print("b1.shape", b1.shape)
        print("b1:  ", b1)
        print()
        print("W2.shape", W2.shape)
        print("W2:  ", W2)
        print()
        print("b2.shape", b2.shape)
        print("b2:  ", b2)
        np.savetxt("rms prop W1.csv", W1, delimiter=',')
        np.savetxt("rms prop b1.csv", b1, delimiter=',')
        np.savetxt("rms prop W2.csv", W2, delimiter=',')
        np.savetxt("rms prop b2.csv", b2, delimiter=',')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
