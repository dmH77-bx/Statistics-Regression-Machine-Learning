# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:18:41 2026

@author: Julia
"""

import numpy as np
import matplotlib.pyplot as plt


def poly_matrix(x, q):
    return np.column_stack([x**k for k in range(q + 1)])

def ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def ridge(X, y, lam):
    n, d = X.shape
    D = np.eye(d)
    D[0, 0] = 0
    return np.linalg.solve(X.T @ X + n * lam * D, X.T @ y)

def train_error(X, y, beta):
    return np.mean((y - X @ beta) ** 2)

def load(name):
    data = np.load(f"../Fichiers_TP/{name}.npy")
    return data[0], data[1]

def fit_and_plot(x, y, q, fig):
    X = poly_matrix(x, q)
    beta = ols(X, y)
    print(f"q = {q}   beta = {beta}   training error = {train_error(X, y, beta):.4f}")
    x_line = np.linspace(x.min(), x.max(), 200)
    plt.figure(fig)
    plt.plot(x, y, 'bx', label="data")
    plt.plot(x_line, poly_matrix(x_line, q) @ beta, 'r-', label=f"OLS q={q}")
    plt.legend()


# Q1 - data1, linear model
x, y = load("data1")
fit_and_plot(x, y, 1, 1)

# Q2 - data2, linear model
x, y = load("data2")
fit_and_plot(x, y, 1, 2)

# Q4 - data2, polynomial of order 2
fit_and_plot(x, y, 2, 3)

# Q5 - data3, vary q
x, y = load("data3")
x_line = np.linspace(x.min(), x.max(), 200)
plt.figure(4)
plt.plot(x, y, 'bx', label="data")
for q in range(1, 16):
    X = poly_matrix(x, q)
    beta = ols(X, y)
    print(f"q = {q:2d}   training error = {train_error(X, y, beta):.4f}")
    if q in (1, 3, 7, 15):
        plt.plot(x_line, poly_matrix(x_line, q) @ beta, label=f"q={q}")
plt.ylim(y.min() - 1, y.max() + 1)
plt.legend()

# Q7 - data3, OLS vs Ridge
q = 12
X = poly_matrix(x, q)
X_line = poly_matrix(x_line, q)

plt.figure(5)
plt.plot(x, y, 'bx', label="data")

beta = ols(X, y)
print(f"OLS          error = {train_error(X, y, beta):.4f}   ||beta|| = {np.linalg.norm(beta[1:]):.2e}")
plt.plot(x_line, X_line @ beta, 'k--', label="OLS")

for lam in [1e-6, 1e-4, 1e-2, 1, 100]:
    beta = ridge(X, y, lam)
    print(f"λ = {lam:8.0e}   error = {train_error(X, y, beta):.4f}   ||beta|| = {np.linalg.norm(beta[1:]):.2e}")
    plt.plot(x_line, X_line @ beta, label=f"Ridge λ={lam:g}")

plt.ylim(y.min() - 1, y.max() + 1)
plt.legend()
plt.show()