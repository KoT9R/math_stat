import math
import numpy as np
import random
import scipy.stats as st
import statistics
import matplotlib.pyplot as plt

n = 50
mu = 0
sigma = 1
low, high = -math.sqrt(3), math.sqrt(3)


def cramerVonMisesTest(x, a, b):
    A = np.ndarray(shape=(n, 1))
    B = np.ndarray(shape=(n, 1))
    C = np.ndarray(shape=(n, 1))
    D = np.ndarray(shape=(n, 1))
    E = np.ndarray(shape=(n, 1))
    F = np.ndarray(shape=(n, 1))
    G = np.ndarray(shape=(n, 1))
    H = np.ndarray(shape=(n, 1))
    I = np.ndarray(shape=(n, 1))

    count1 = 0
    count2 = 0
    for j in range(n):
        if (x[j] > b):
            count1 = count1 + 1

    for j in range(n):
        if (x[j] < a):
            count2 = count2 + 1
    print("count1 = " + str(count1))
    print("count2 = " + str(count2))

    for j in range(n):
        A[j] = (2 * j + 1) / (2 * n)

        if (x[j] < a):
            B[j] = 0
        else:
            if (x[j] > b):
                B[j] = 1
            else:
                B[j] = (x[j] - a) / (b - a)

    C[j] = math.log(B[j])
    D[j] = A[j] * C[j]
    E[j] = 1 - A[j]
    F[j] = 1 - B[j]
    G[j] = math.log(F[j])
    H[j] = E[j] * G[j]
    I[j] = D[j] + H[j]

    print(np.sum(I))
    stat = -n - 2 * np.sum(I)

    for j in range(n):
        print("%.2s & " % str(j + 1),
              "%s & " % str("%.3f" % A[j]),
              "%s & " % str("%.6f" % B[j]),
              "%s & " % str("%.5f" % C[j]),
              "%s & " % str("%.5f" % D[j]),
              "%s & " % str("%.4f" % E[j]),
              "%s & " % str("%.6f" % F[j]),
              "%s & " % str("%.5f" % G[j]),
              "%s & " % str("%.5f" % H[j]),
              "%s \\\\" % str("%.5f" % I[j]),
              "\hline"
              )
    return stat


def main():
    data = np.random.uniform(low, high, n)
    data = np.sort(data)

    x_max = max(data)
    x_min = min(data)
    a = x_min - (x_max + x_min) / (n - 1)
    b = x_max + (x_max - x_min) / (n - 1)
    print("Оценки параметров:" + str("%.4f" % a), str("%.4f" % b))

    count, bins, ignored = plt.hist(data, 30, density=True)
    plt.xlabel("")
    plt.ylabel("frequency")
    plt.show()

    for j in range(n):
        print("%s &" % str(j + 1),
              "%s \\\\" % data[j],
              "\hline"
              )

    stat = cramerVonMisesTest(data, a, b)
    print("Значение статистики:" + str(stat))


if __name__ == "__main__":
    main()
