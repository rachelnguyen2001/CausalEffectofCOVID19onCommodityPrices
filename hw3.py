#!/usr/bin/env python

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np


def ipw(Y, A, Z, data, trim):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via IPW

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    formula = A + '~'

    for i in range(len(Z)):
        formula += ' + '
        formula += Z[i]

    model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial()).fit()
    # X = pd.DataFrame(data.drop([A], axis=1))
    # y = model.predict(X)
    data["ps"] = model.predict(data)
    # print(data["ps"])
    # print(y)
    # print(propensity.head())
    # print(model.predict(data))

    if trim:
        data = data.query('ps >= 0.1 and ps <= 0.9')
        # print(data)

    n = data.shape[0]
    ACE = 0.0
    
    for i in range(n):
        first = (data.iloc[i][A] * data.iloc[i][Y]) / data.iloc[i]["ps"]
        second = ((1 - data.iloc[i][A]) * data.iloc[i][Y]) / (1 - data.iloc[i]["ps"])
        ACE += (first - second)
    
    ACE /= n
    # print(ACE)
    
    return ACE

def augmented_ipw(Y, A, Z, data, trim):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via AIPW

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not

    Return
    ------
    ACE: float corresponding to the causal effect
    """
    f_1 = A + '~'
    f_2 = Y + '~' + A

    for i in range(len(Z)):
        f_1 += ' + '
        f_2 += ' + '
        f_1 += Z[i]
        f_2 += Z[i]

    model_1 = sm.GLM.from_formula(f_1, data=data, family=sm.families.Binomial()).fit()
    model_2 = smf.ols(formula=f_2, data=data).fit()
    D_a = data.copy(deep=True)
    D_a_prime = data.copy(deep=True)
    D_a[A].replace({0: 1}, inplace=True)
    D_a_prime[A].replace({1:0}, inplace=True)
    data["D_a_predict"] = model_2.predict(D_a)
    data["D_a_prime_predict"] = model_2.predict(D_a_prime)
    # X = pd.DataFrame(data.drop([A], axis=1))
    data["ps"] = model_1.predict(data)
    data["Y | A, Z"] = model_2.predict(data)
    
    ACE = 0.0

    if trim:
        data = data.query('ps >= 0.1 and ps <= 0.9')
    
    n = data.shape[0]

    for i in range(n):
        f_ipw = (data.iloc[i][A] * data.iloc[i][Y]) / data.iloc[i]["ps"]
        s_ipw = ((1 - data.iloc[i][A]) * data.iloc[i][Y]) / (1 - data.iloc[i]["ps"])
        ipw = f_ipw - s_ipw
        f_bd = data.iloc[i]["D_a_predict"]
        s_bd = data.iloc[i]["D_a_prime_predict"]
        bd = f_bd - s_bd
        f_ag = (data.iloc[i][A] * data.iloc[i]["Y | A, Z"]) / data.iloc[i]["ps"]
        s_ag = ((1 - data.iloc[i][A]) * data.iloc[i]["Y | A, Z"]) / (1 - data.iloc[i]["ps"])
        ag = f_ag - s_ag
        val = ipw + bd - ag
        ACE += val

    ACE /= n
    return ACE

def compute_confidence_intervals(Y, A, Z, data, method_name, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for IPW or AIPW (potentially with trimming) via bootstrap.
    The input method_name can be used to decide how to compute the confidence intervals.

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    # code for bootstrap

    return -1, 1


def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """

    np.random.seed(100)

    nsw_observational = pd.read_csv("nsw_observational.txt")
    # define some backdoor sets
    Z = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

    # point estimate and CIs using observational data and IPW
    print("ACE using IPW", 0, (-1, 1))
    print(ipw("re78", "treat", Z, nsw_observational, False))

    # point estimate and CIs using observational data and IPW with trimming=
    print("ACE using IPW with trimming", 0, (-1, 1))
    print(ipw("re78", "treat", Z, nsw_observational, True))

    # point estimate and CIs using observational data and AIPW
    print("ACE using AIPW (potentially with trimming)", (0, -1, 1))
    print(augmented_ipw("re78", "treat", Z, nsw_observational, True))


if __name__ == "__main__":
    main()
