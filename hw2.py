#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np

def backdoor_adjustment(Y, A, Z, data):
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

    ACE = 0
    return ACE


def compute_confidence_intervals(Y, A, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):

        # Implement your code here:
        pass

    q_low, q_up = -1, 1
    return q_low, q_up


def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """

    np.random.seed(0)

    nsw_randomized = pd.read_csv("nsw_randomized.txt")
    nsw_observational = pd.read_csv("nsw_observational.txt")

    # point estimate and CIs from the randomized trial
    print("ACE from randomized trial", 0, (-1, 1))

    # point estimate and CIs from unadjusted observational data
    print("ACE without adjustment in observational data", 0, (-1, 1))

    # point estimate and CIs using observational data and linear regression
    print("ACE with adjustment in observational data", 0, (-1, 1))

    # point estimate and CIs using observational data and nonlinear regression
    print("ACE with adjustment in a non-linear regression", 0, (-1, 1))


if __name__ == "__main__":
    main()
