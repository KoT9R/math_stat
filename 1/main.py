import numpy as np
import matplotlib.pyplot as plt

POISSON_PARAM = 3
UNIFORM_FRONT = np.sqrt(3)


def normalized_distribution(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)


def laplace_distribution(x):
    return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(x))


def uniform_distribution(x):
    flag = (x <= UNIFORM_FRONT)
    return 1 / (2 * UNIFORM_FRONT) * flag


def cauchy_distribution(x):
    return 1 / (np.pi * (1 + x * x))


def poisson_distribution(x):
    k = POISSON_PARAM
    return (np.power(x, k) / np.math.factorial(k)) * np.exp(-x)


func_dict = {
    'normal':   normalized_distribution,
    'laplace':  laplace_distribution,
    'uniform':  uniform_distribution,
    'cauchy':   cauchy_distribution,
    'poisson':  poisson_distribution,
}


def generate_laplace(x):
    return np.random.laplace(0, 1/np.sqrt(3), x)


def generate_uniform(x):
    return np.random.uniform(-UNIFORM_FRONT, UNIFORM_FRONT, x)


def generate_poisson(x):
    return np.random.poisson(POISSON_PARAM, x)


generate_dict = {
    'normal':   np.random.standard_normal,
    'laplace':  generate_laplace,
    'uniform':  generate_uniform,
    'cauchy':   np.random.standard_cauchy,
    'poisson':  generate_poisson,
}


def draw(array, func, sector):
    if sector == 3:
        plt.title('Histograms for 20, 100, 500 elements. Distribution: ' + func)
    plt.subplot(130+sector)
    plt.hist(array, 10, density=True)
    xx = np.linspace(np.min(array), np.max(array), 100)
    plt.plot(xx, func_dict[func](xx), 'r')


def research(distribution_type):
    plt.figure("distribution " + distribution_type)
    num = 20
    sector = 1
    for i in range(3):
        draw(generate_dict[distribution_type](num), distribution_type, sector)
        num *= 5
        sector += 1
    plt.show()


if __name__ == "__main__":

    research('normal')
    research('laplace')
    research('cauchy')
    research('poisson')
    research('uniform')



