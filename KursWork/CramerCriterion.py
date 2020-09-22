import math
import numpy as np
import random
import scipy.stats as st
import statistics

n = 50
mu = 0
sigma = 1


def cramerVonMisesTest(x, mu_estimate, sigma_estimate):
    A = np.ndarray(shape=(n, 1))
    B = np.ndarray(shape=(n, 1))
    C = np.ndarray(shape=(n, 1))
    D = np.ndarray(shape=(n, 1))
    E = np.ndarray(shape=(n, 1))
    F = np.ndarray(shape=(n, 1))
    G = np.ndarray(shape=(n, 1))
    H = np.ndarray(shape=(n, 1))
    I = np.ndarray(shape=(n, 1))

    for j in range(n):
        A[j] = (2 * j + 1) / (2 * n)
        B[j] = st.norm.cdf(x[j], loc=mu_estimate, scale=sigma_estimate)
        C[j] = math.log(B[j])
        D[j] = A[j] * C[j]
        E[j] = 1 - A[j]
        F[j] = 1 - B[j]
        G[j] = math.log(F[j])
        H[j] = E[j] * G[j]
        I[j] = D[j] + H[j]

    stat = -n - 2 * np.sum(I)

    for j in range(n):
        print("%2s & " % str(j + 1),
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
    x = np.random.normal(mu, sigma, n)
    x = np.sort(x)

    mu_estimate = np.mean(x)
    sigma_estimate = math.sqrt(statistics.variance(x))
    print("Метод максимального правдоподобия:" + str("%.4f" % mu_estimate), str("%.4f" % sigma_estimate))

    stat = cramerVonMisesTest(x, mu_estimate, sigma_estimate)
    print("Значение статистики:" + str(stat))


if __name__ == "__main__":
    main()
