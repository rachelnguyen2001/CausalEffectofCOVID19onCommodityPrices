#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.special import expit

def odds_ratio(X, Y, Z, data):
    """
    Compute the odds ratio OR(X, Y | Z).
    X, Y are names of variables
    in the data frame. Z is a list of names of variables.

    Return float OR for the odds ratio OR(X, Y | Z)
    """

    # Formula for the model
    formula = Y + ' ~ ' + X

    # Adding variables to the formula
    for i in range(len(Z)):
        formula += '+'
        formula += Z[i]

    # The model
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial()).fit()

    # Odds ratio is the exponent of the coefficient of X
    OR = np.exp(model.params[X])
    
    return OR

def compute_confidence_intervals(X, Y, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals through bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []  

    # Generate datasets by resampling with replacement and calculate theta values
    for i in range(num_bootstraps):
        theta = odds_ratio(X, Y, Z, data.sample(n = data.shape[0], replace=True))
        estimates.append(theta)    

    # Construct a Series
    series = pd.Series(estimates)

    # Get values at the Ql and Qu quantiles
    q_low, q_up = series.quantile(Ql), series.quantile(Qu)    

    return q_low, q_up

def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(200)
    data = pd.read_csv("data.txt") 
    print(odds_ratio("opera", "mortality", [], data), compute_confidence_intervals("opera", "mortality", [], data))
    print(odds_ratio("opera", "mortality", ["income"], data),
          compute_confidence_intervals("opera", "mortality", ["income"], data))

    print(odds_ratio("mortality", "opera", [], data), compute_confidence_intervals("mortality", "opera", [], data))
    print(odds_ratio("mortality", "opera", ["income"], data),
          compute_confidence_intervals("mortality", "opera", ["income"], data))

if __name__ == "__main__":
    main()
