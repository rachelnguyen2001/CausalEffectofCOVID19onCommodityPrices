#!/usr/bin/env python

import statsmodels.api as sm
import statsmodels.formula.api as smf
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

    formula = Y + ' ~ ' + A

    for i in range(len(Z)):
        formula += ' + '
        formula += Z[i]
        
    model = smf.ols(formula=formula, data=data).fit()

    D_a = data.copy(deep=True)
    D_a_prime = data.copy(deep=True)
    D_a[A].replace({0: 1}, inplace=True)
    D_a_prime[A].replace({1:0}, inplace=True)
    
    ACE = 0.0
    n = data.shape[0]
    # print(n)
    
    # print(model.predict(D_a)[15])
    p_1 = model.predict(D_a)
    p_2 = model.predict(D_a_prime)

    # print(p_1)
    # print(p_2)
    
    for i in range(n):
        # print(model.predict(D_a.iloc[i]))
        ACE += p_1.iloc[i]
        ACE -= p_2.iloc[i]
        # print(ACE)
        # ACE -= model.predict(D_a_prime.iloc[i])

    ACE /= n
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
        val = backdoor_adjustment(Y, A, Z, data.sample(n = data.shape[0], replace=True))
        # print(val)
        estimates.append(val)

    series = pd.Series(estimates)
    q_low, q_up = series.quantile(Ql), series.quantile(Qu)
    return q_low, q_up


def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """

    np.random.seed(0)

    nsw_randomized = pd.read_csv("nsw_randomized.txt")
    nsw_observational = pd.read_csv("nsw_observational.txt")
    Y = "re78"
    A = "treat"
    Z = []

    # point estimate and CIs from the randomized trial
    print("ACE from randomized trial", backdoor_adjustment(Y, A, Z, nsw_randomized), compute_confidence_intervals(Y, A, Z, nsw_randomized))

    # point estimate and CIs from unadjusted observational data
    print("ACE without adjustment in observational data", backdoor_adjustment(Y, A, Z, nsw_observational), compute_confidence_intervals(Y, A, Z, nsw_observational))

    Z = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
    # point estimate and CIs using observational data and linear regression
    print("ACE with adjustment in observational data", backdoor_adjustment(Y, A, Z, nsw_observational), compute_confidence_intervals(Y, A, Z, nsw_observational))

    Z.append("black*nodegree")
    Z.append("black*treat")
    Z.append("black*educ")
    Z.append("hisp*nodegree")
    Z.append("hisp*treat")
    Z.append("hisp*educ")
    Z.append("treat*nodegree")
    Z.append("treat*educ")
    # point estimate and CIs using observational data and nonlinear regression
    print("ACE with adjustment in a non-linear regression", backdoor_adjustment(Y, A, Z, nsw_observational), compute_confidence_intervals(Y, A, Z, nsw_observational))


if __name__ == "__main__":
    main()
