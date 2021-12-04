#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt

def backdoor_ML(Y, A, Z, data, cases):
    regressors_y = tuple([A] + Z)
    X = np.column_stack(regressors_y)
    model = RandomForestRegressor().fit(X, Y)
    data_A = X.copy()
    data_A[:,0] = cases
    return np.mean(model.predict(data_A))

def backdoor_adjustment(Y, A, Z, data, cases):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    formula: list of variable names included the backdoor adjustment set
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    # Formula for the model
    formula = Y + ' ~ ' + A

    # Adding variables to the formula
    for i in range(len(Z)):
        formula += ' + '
        formula += Z[i]

    # The model
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()

    # Create two copies of the data
    D_a = data.copy(deep=True)
    # D_a_prime = data.copy(deep=True)
    D_a[A] = cases
    # Set A = 1 for all data points in D_a and A = 0 for all data points in D_a_prime
    # D_a[A].replace({0: 1}, inplace=True)
    # D_a_prime[A].replace({1:0}, inplace=True)
    
    ACE = 0.0

    # Number of samples
    n = data.shape[0]
    
    # Prediction for Y in D_a and D_a_prime
    D_a_predict = model.predict(D_a)
    # D_a_prime_predict = model.predict(D_a_prime)

    # Compute the mean difference in predicted potential outcomes
    # for i in range(n):
        # ACE += (D_a_predict.iloc[i] - D_a_prime_predict.iloc[i])

    ACE = np.mean(D_a_predict)
    
    return ACE


def compute_confidence_intervals(Y, A, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    n = data.shape[0]

    # Generate datasets by resampling with replacement and calculate estimates
    for i in range(num_bootstraps):
        estimate = backdoor_adjustment(Y, A, Z, data.sample(n = n, replace=True))
        estimates.append(estimate)

    # Construct a Series
    series = pd.Series(estimates)

    # Get values at the Ql and Qu quantiles
    q_low, q_up = series.quantile(Ql), series.quantile(Qu)

    return q_low, q_up

def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """

    np.random.seed(0)

    data = pd.read_csv("vaccine_estimate.csv")
    data = data.dropna()
    # print(data)
    Y = "crudeoil"
    A = "cases_three"
    Z = ["cases"]
    # print("ACE:", backdoor_adjustment(Y, A, Z, data, 295883))
    # print("ACE_ML:", backdoor_ML(data[Y], data[A], [data["cases"]], data, 295883))
    x = []
    y = []

    for i in range(5700, 305000, 2000):
        x.append(i)
        # y.append(backdoor_adjustment(Y, A, Z, data, i))
        y.append(backdoor_ML(data[Y], data[A], [data["cases"]], data, i))

    # print(x)
    print(y)
    # plt.plot(x, y)
    # plt.show()
    # print("ACE_ML:", backdoor_ML(data[Y], data[A], [data["cases"]], data, 295883))
    # print(len(data.columns))
    # print(data.shape[0])
    # print(data.head())
    # Y = "crudeoil_five"
    # A = "cases_three"
    # Z = ["cases_two"]
    # print(data[A])
    # print("ACE:", backdoor_adjustment(Y, A, Z, data))

if __name__ == "__main__":
    main()
