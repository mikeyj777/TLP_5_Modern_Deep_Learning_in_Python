import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import pandas as pd

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_b1, derivative_b2, derivative_w1, derivative_w2


def main():

    max_iter = 20  # make it 30 for sigmoid
    print_period = 50

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

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

    # 1st moment
    mW1 = 0
    mb1 = 0
    mW2 = 0
    mb2 = 0

    # 2nd moment
    vW1 = 0
    vb1 = 0
    vW2 = 0
    vb2 = 0

    # hyperparameters
    lr0 = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    losses_adam = []
    errors_adam = []

    t = 1

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
            gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1

            # new m
            mW1 = beta1 * mW1 + (1 - beta1) * gW1
            mb1 = beta1 * mb1 + (1 - beta1) * gb1
            mW2 = beta1 * mW2 + (1 - beta1) * gW2
            mb2 = beta1 * mb2 + (1 - beta1) * gb2

            # new v
            vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
            vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
            vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2

            # bias correction
            correction1 = 1 - beta1 ** t
            hat_mW1 = mW1 / correction1
            hat_mb1 = mb1 / correction1
            hat_mW2 = mW2 / correction1
            hat_mb2 = mb2 / correction1

            correction2 = 1 - beta2 ** t
            hat_vW1 = vW1 / correction2
            hat_vb1 = vb1 / correction2
            hat_vW2 = vW2 / correction2
            hat_vb2 = vb2 / correction2

            # update t
            t += 1

            # apply updates to the params
            W1 = W1 - lr0 * hat_mW1 / np.sqrt(hat_vW1 + eps)
            b1 = b1 - lr0 * hat_mb1 / np.sqrt(hat_vb1 + eps)
            W2 = W2 - lr0 * hat_mW2 / np.sqrt(hat_vW2 + eps)
            b2 = b2 - lr0 * hat_mb2 / np.sqrt(hat_vb2 + eps)

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_adam.append(l)
                print("cost at iter i %d, j %d:  %.6f" % (i, j, l))

                e = error_rate(pY, Ytest)
                errors_adam.append(e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("final error rate:  ", error_rate(pY, Ytest))
    print()
    plt.plot(losses_adam, label='adam')

    pY, _ = forward(Xtest, W1, b1, W2, b2)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
