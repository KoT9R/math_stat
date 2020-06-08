import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def fit(X, params):
    return X.dot(params)


def cost_function(params, X, y):
    return np.sum(np.abs(y - fit(X, params)))


def generate(a, b, h):
    n = int((b - a) / h) + 2
    x = np.linspace(-1.8, 2, n)
    e_i = np.random.standard_normal(n)
    y = 2 + 2 * x + e_i
    return x, y


def perturbation(y):
    res = []
    for i in range(0, len(y)):
        res.append(y[i])
    back = len(y) - 1
    res[0] = y[0] + 10
    res[back] = y[back] - 10
    return res


def mnk(x, y):
    a2 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    b2 = np.mean(y) - a2 * np.mean(x)
    return a2, b2


def mnm(x, y):
    X = np.asarray([np.ones(20, ), x]).T
    x0 = np.ones(2)
    output = minimize(cost_function, x0, args=(X, y))

    y_hat = fit(X, output.x)

    print("y hat = ", np.median(y_hat))

    return x, y_hat


def dist(a, b, x, y):
    return np.abs(a * x - 1 * y + b) / np.sqrt(np.power(a, 2) + 1)


def deviation(l, r, h, N):
    mnk_sum_q, mnk_sum_q_per = 0, 0
    mnm_sum_q, mnm_sum_q_per = 0, 0
    x = 0
    for i in range(0, N):
        x, y = generate(l, r, h)
        mnk_a, mnk_b = mnk(x, y)
        mnm_a, mnm_b = mnm(x, y)

        mnk_q = 1 / len(x) * np.sum(dist(mnk_a, mnk_b, x, y))
        mnm_q = 1 / len(x) * np.sum(dist(mnm_a, mnm_b, x, y))

        mnk_sum_q += mnk_q
        mnm_sum_q += mnm_q

        x, y = generate(l, r, h)
        y = perturbation(y)
        mnk_a, mnk_b = mnk(x, y)
        mnm_a, mnm_b = mnm(x, y)

        mnk_q_per = 1 / len(x) * np.sum(dist(mnk_a, mnk_b, x, y))
        mnm_q_per = 1 / len(x) * np.sum(dist(mnm_a, mnm_b, x, y))

        mnk_sum_q_per += mnk_q_per
        mnm_sum_q_per += mnm_q_per

    print("           mnm             mnk")
    print("original ", mnm_sum_q / N, mnk_sum_q / N)
    print("distorted", mnm_sum_q_per / N, mnk_sum_q_per / N)


if __name__ == "__main__":
    l = -1.8
    r = 2
    h = 0.2
    a = 2
    b = 2

    deviation(l, r, h, 1000)

    plt.figure()
    plt.subplot(121)
    plt.title("Original sample")
    print("\t\t\tOriginal sample")

    x, y = generate(l, r, h)
    print("%12s:\t a = %lf, b = %lf" % ("Model sample", a, b))
    plt.plot(x, a * x + b, 'b', label='Model sample')

    plt.scatter(x, y)

    m, c = mnk(x, y)
    print("%12s:\ta = %lf, b = %lf" % ("МНК", m, c))
    plt.plot(x, m * x + c, 'r', label='МНК')

    m, c = mnm(x, y)
    plt.plot(m, c, 'g', label='МНМ')

    plt.legend()

    print("\n")
    plt.subplot(122)
    plt.title("Distorted sample")
    print("\t\t\tDistorted sample")

    x, y = generate(l, r, h)
    y = perturbation(y)

    print("%12s:\ta = %lf, b = %lf" % ("Model sample", a, b))
    plt.plot(x, a * x + b, 'b', label='Model sample')
    plt.scatter(x, y)

    m, c = mnk(x, y)
    print("%12s:\ta = %lf, b = %lf" % ("МНК", m, c))
    plt.plot(x, m * x + c, 'r', label='МНК')

    xx, yy = mnm(x, y)
    plt.plot(xx, yy, 'g', label='МНМ')

    plt.legend()
    plt.show()
