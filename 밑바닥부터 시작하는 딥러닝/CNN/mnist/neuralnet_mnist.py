import pickle

import numpy as np

from dataset.mnist import load_mnist
import activation_function as af

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("./dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = af.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = af.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = af.softmax(a3)

    return y

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error_ohe(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# t가 라벨값인 경우의 크로스 엔트로피 함수
def cross_entropy_error_lbl(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#                         ---------------------------
#                        np.arange(batch_size) 함수는 0 ~ batch_size-1까지의 배열을 만들어
#                        이 부분은 y의 배치 사이즈 순번에서 t를 인덱스로 하는 값을 가져와 처리한다.

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

# function_2 함수에서 x0 = 3, x1 = 4일 때, x0에 대한 편미분을 구하기 위한 함수
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0

# function_2 함수에서 x0 = 3, x1 = 4일 때, x1에 대한 편미분을 구하기 위한 함수
def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


numerical_diff(function_tmp1, 3.0)

# import matplotlib.pylab as plt

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("X")
# plt.ylabel("f(x)")
# plt.plot(x, y)
#
# tf = tangent_line(function_1, 5)
# y2 = tf(x)
#
# plt.plot(x, y2)
# plt.show()
#
# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))