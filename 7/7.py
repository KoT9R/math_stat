import numpy as np
from tabulate import tabulate
import scipy.stats as stats

start_border, end_border = -2.0, 2.0
sample_size = 100
alpha = 0.05
p = 1 - alpha
k = 7


def MLE(sample):
    mu_ml = np.mean(sample)
    sigma_ml = np.std(sample)
    print("mu_ml = ", np.around(mu_ml, decimals=4),
          " sigma_ml=", np.around(sigma_ml, decimals=4))
    return mu_ml, sigma_ml


def quantileChi2(sample, mu, sigma):
    hypothesis = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)

    borders = np.linspace(start_border, end_border, num=k - 1)

    probabilities = np.array(hypothesis(start_border))
    quantities = np.array(len(sample[sample < start_border]))

    for i in range(k - 2):
        p_i = hypothesis(borders[i + 1]) - hypothesis(borders[i])
        probabilities = np.append(probabilities, p_i)
        n_i = len(sample[(sample < borders[i + 1]) & (sample >= borders[i])])
        quantities = np.append(quantities, n_i)

    probabilities = np.append(probabilities, 1 - hypothesis(end_border))
    quantities = np.append(quantities, len(sample[sample >= end_border]))

    chi2 = np.divide(
        np.multiply(
            (quantities - sample_size * probabilities),
            (quantities - sample_size * probabilities)
        ),
        probabilities * sample_size
    )

    quantile = stats.chi2.ppf(p, k - 1)
    print("Quantile:", quantile)
    return chi2, borders, probabilities, quantities


def buildTable(chi2, borders, probabilities, quantities):
    headers = ["i", "interval", "n_i", "p_i",
               "(n_i - np_i)^2/np_i"]
    rows = []
    for i in range(0, len(quantities)):
        if i == 0:
            limits = ["infty", np.around(borders[0], decimals=2)]
        elif i == len(quantities) - 1:
            limits = [np.around(borders[-1], decimals=2), "infty"]
        else:
            limits = [np.around(borders[i - 1], decimals=2), np.around(borders[i], decimals=2)]
        rows.append(
            [i + 1,
             limits,
             quantities[i],
             np.around(probabilities[i], decimals=4),
             np.around(chi2[i], decimals=4)]
        )
    rows.append(["sum", "--",
                 np.sum(quantities),
                 np.around(np.sum(probabilities), decimals=4),
                 np.around(np.sum(chi2), decimals=4)]
                )
    return tabulate(rows, headers)


def kek():
    distr = np.random.normal(0, 1, size=20)
    mu_n = np.mean(distr)
    sigma_n = np.std(distr)
    print(np.around(mu_n, decimals=2), ' ', np.around(sigma_n, decimals=2))

    alpha = 0.05
    p = 1 - alpha
    k = 7

    limits = np.linspace(-2, 2, num=k - 1)
    sample = stats.chi2.ppf(p, k - 1)
    array = np.array([stats.norm.cdf(limits[0])])
    quan_ar = np.array([len(distr[distr <= limits[0]])])
    for i in range(0, len(limits) - 1):
        new_ar = stats.norm.cdf(limits[i + 1]) - stats.norm.cdf(limits[i])
        array = np.append(array, new_ar)
        quan_ar = np.append(quan_ar, len(distr[(distr <= limits[i + 1]) & (distr >= limits[i])]))
    array = np.append(array, 1 - stats.norm.cdf(limits[-1]))
    quan_ar = np.append(quan_ar, len(distr[distr >= limits[-1]]))
    result = np.divide(np.multiply((quan_ar - 20 * array), (quan_ar - 20 * array)), array * 20)

    headers = ["i", "interval", "n_i", "p_i", "(n_i - np_i)^2/np_i"]
    rows = []
    for i in range(0, len(quan_ar)):
        if i == 0:
            boarders = ['-inf', np.around(limits[0], decimals=4)]
        elif i == len(quan_ar) - 1:
            boarders = [np.around(limits[-1], decimals=4), 'inf']
        else:
            boarders = [np.around(limits[i - 1], decimals=4), np.around(limits[i], decimals=4)]
        rows.append(
            [i + 1, boarders,
            quan_ar[i],
            np.around(array[i], decimals=4),
            np.around(result[i], decimals=4)])
    rows.append(["sum", "--",
                 np.sum(quan_ar),
                 np.around(np.sum(array), decimals=4),
                 np.around(np.sum(result), decimals=4)])
    print(tabulate(rows, headers))

    print(len(quan_ar))
    print('\n')


if __name__ == '__main__':
    normal_sample = np.random.normal(0, 1, size=sample_size)
    mu_ml, sigma_ml = MLE(normal_sample)
    chi2, borders, probabilities, quantities = quantileChi2(normal_sample, mu_ml, sigma_ml)
    print(buildTable(chi2, borders, probabilities, quantities))

    kek()
