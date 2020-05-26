import numpy as np
from prettytable import PrettyTable

UNIFORM_FRONT = np.sqrt(3)
TESTS_NUM = 1000
TRUNCATION = 0.25
POISSON_PARAM = 3


def generate_laplace(x):
    return np.random.laplace(0, 1 / np.sqrt(3), x)


def generate_uniform(x):
    return np.random.uniform(-UNIFORM_FRONT, UNIFORM_FRONT, x)


def generate_poisson(x):
    return np.random.poisson(POISSON_PARAM, x)


generate_dict = {
    'normal': np.random.standard_normal,
    'laplace': generate_laplace,
    'uniform': generate_uniform,
    'cauchy': np.random.standard_cauchy,
    'poisson': generate_poisson
}


def find_sample_mean(sample):
    return np.mean(sample)


def find_median(sample):
    return np.median(sample)


def find_half_sum_extreme(sample):
    return (min(sample) + max(sample)) / 2


def find_quantile(sample, index):
    return np.quantile(sample, index)


def find_half_sum_quantile(sample):
    return (find_quantile(sample, 0.25) + find_quantile(sample, 0.75)) / 2


def find_truncated_mean(sample):
    res = 0
    n = len(sample)
    r = int(TRUNCATION * n)
    i = r + 1
    while i <= n - r:
        res += sample[i]
        i = i + 1
    return res / (n - 2 * r)


characteristic_dict = {
    'sample_mean': find_sample_mean,
    'med': find_median,
    'half_sum_extreme': find_half_sum_extreme,
    'half_sum_quantile': find_half_sum_quantile,
    'truncated_mean': find_truncated_mean,
}


def research(distribution_type, characteristic_type):
    res = []
    num = 20
    values = [0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(TESTS_NUM):
            sample = np.sort(generate_dict[distribution_type](num))
            res.append(characteristic_dict[characteristic_type](sample))
            values[i] = np.mean(res)
            values[i + 3] = D(res)
        res = []
        num += 40
    return values


def D(sample):
    return np.var(sample)


def print_table(distribution_type):
    s = research(distribution_type, 'sample_mean')
    m = research(distribution_type, 'med')
    zr = research(distribution_type, 'half_sum_extreme')
    zq = research(distribution_type, 'half_sum_quantile')
    ztr = research(distribution_type, 'truncated_mean')
    precision = 10000
    if distribution_type == 'cauchy':
        precision = 1
    for i in range(len(s)):
        s[i] = int(s[i] * precision) / precision
        m[i] = int(m[i] * 10000) / 10000
        zr[i] = int(zr[i] * precision) / precision
        zq[i] = int(zq[i] * 10000) / 10000
        ztr[i] = int(ztr[i] * 10000) / 10000
    x = PrettyTable()

    x.field_names = [distribution_type, "sample_mean", "med_x", "Z_R", "Z_Q", "Z_tr"]

    x.add_row(["E(z) 20", s[0], m[0], zr[0], zq[0], ztr[0]])
    x.add_row(["D(z) 20", s[3], m[3], zr[3], zq[3], ztr[3]])
    x.add_row(["E(z) 60", s[1], m[1], zr[1], zq[1], ztr[1]])
    x.add_row(["D(z) 60", s[4], m[4], zr[4], zq[4], ztr[4]])
    x.add_row(["E(z) 100", s[2], m[2], zr[2], zq[2], ztr[2]])
    x.add_row(["D(z) 100", s[5], m[5], zr[5], zq[5], ztr[5]])

    print(distribution_type, "sample_mean", "med_x", "Z_R", "Z_Q", "Z_tr")
    print("E(z)20", s[0], m[0], zr[0], zq[0], ztr[0])
    print("D(z)20", s[3], m[3], zr[3], zq[3], ztr[3])
    print("E(z)60", s[1], m[1], zr[1], zq[1], ztr[1])
    print("D(z)60", s[4], m[4], zr[4], zq[4], ztr[4])
    print("E(z)100", s[2], m[2], zr[2], zq[2], ztr[2])
    print("D(z)100", s[5], m[5], zr[5], zq[5], ztr[5])
    print(x)


if __name__ == "__main__":
    print_table('normal')
    print_table('cauchy')
    print_table('laplace')
    print_table('uniform')
    print_table('poisson')
