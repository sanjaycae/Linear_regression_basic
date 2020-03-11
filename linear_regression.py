import numpy as np
import matplotlib.pyplot as plt  # To visualize


def mean(z):
    return sum(z) / len(z)


def b1_func(x_, y_):
    xy = [i * j for i, j in zip(x_, y_)]
    y_mean_x = [mean(y_) * i for i in x_]
    x2 = np.square(x_)
    x_mean_x = [mean(x_) * i for i in x_]
    numerator = [i - j for i, j in zip(xy, y_mean_x)]
    denominator = [i - j for i, j in zip(x2, x_mean_x)]
    return sum(numerator) / sum(denominator)


def b0_func(x_, y_, b1_):
    x_mean = sum(x_) / len(x_)
    y_mean = sum(y_) / len(y_)
    return y_mean - (b1_ * x_mean)


def y_pred_func(b0_, b1_, x_):
    return [b0_ + (b1_ * i) for i in x_]


def mse_func(y_, y_pred_):
    error = [(i - j) ** 2 for i, j in zip(y_, y_pred_)]
    return mean(error)


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [3, 2, 2, 4, 3]
    print("INFO X ::", x)
    print("INFO Y ::", y)
    b1 = b1_func(x, y)
    print("bi  INFO ::", b1)
    b0 = b0_func(x, y, b1)
    print("b0 INFO ::", b0)
    y_pred = y_pred_func(b0, b1, x)
    print("Y Predicted :", y_pred)
    mse = mse_func(y, y_pred)
    print("Mean square error :", mse)

    plt.scatter(x,y)
    plt.plot(x, y_pred, color='red')
    plt.show()
