import numpy as np
import seaborn as sns
import matplotlib.pyplot as mplot

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


def IQR(x):
    return np.abs(np.quantile(x, 1 / 4) - np.quantile(x, 3 / 4))


# noq: 3
def test(x: int, y):
    z: float
    k: float
    k = 5
    y: str = "a"
    z = x + y
    x: float = 1.2
    while z < 4:
        print(z)
        z += 1
    return x


def outlier(sample):
    length = sample.size
    res = 0
    left = np.quantile(sample, 1 / 4) - 1.5 * IQR(sample)
    right = np.quantile(sample, 3 / 4) + 1.5 * IQR(sample)
    for i in range(0, length):
        if sample[i] < left or sample[i] > right:
            res += 1
    return res / length


def research(dist_type):
    values = []
    variance = []
    for j in range(2):
        res = 0
        for i in range(0, TESTS_NUM):
            sample = np.sort(generate_dict[dist_type](100 - 80 * j))
            res += outlier(sample)
        values.append(res / TESTS_NUM)
        varis = [outlier(np.sort(generate_dict[dist_type](100 - 80 * j))) for i in range(0, 1000)]
        variance.append(np.var(varis))
    print(values)
    print(dist_type)

    print("n = 20", values[0])
    print("D =", variance[0])
    print("n = 100", values[1])
    print("D =", variance[1])

    data = []
    for i in range(2):
        data.append(np.sort(generate_dict[dist_type](100 - i * 40)))

    mplot.figure(dist_type)
    mplot.title(dist_type + " distribution")
    sns.set(style="whitegrid")
    sns.boxplot(data=data, orient='h')
    mplot.yticks(np.arange(2), ('100', '20'))
    mplot.show()
    return values


if __name__ == "__main__":
    # research('normal')
    # research('cauchy')
    # research('laplace')
    # research('uniform')
    research('poisson')
