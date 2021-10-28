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

    # Formula for the model
    formula = A + '~'

    # Adding variables to the formula
    for i in range(len(Z)):
        formula += ' + '
        formula += Z[i]

    # The model
    model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial()).fit()
    # Propensity scores
    data["propensity_score"] = model.predict(data)

    # Trimming the data
    if trim:
        data = data.query('propensity_score >= 0.1 and propensity_score <= 0.9')

    ACE = np.mean((data[A] * data[Y]) / data["propensity_score"] - ((1 - data[A]) * data[Y]) / (1 - data["propensity_score"]))
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
    # Formula for the propensity score model
    propensity_formula = A + '~'
    # Formula for the outcome regression model
    regression_formula = Y + '~' + A

    # Adding variables to the formulas
    for i in range(len(Z)):
        propensity_formula += ' + '
        regression_formula += ' + '
        propensity_formula += Z[i]
        regression_formula += Z[i]

    # The propensity score model
    propensity_model = sm.GLM.from_formula(propensity_formula, data=data, family=sm.families.Binomial()).fit()
    # The outcome regression model
    regression_model = smf.ols(formula=regression_formula, data=data).fit()

    # Create two copies of the data
    D_a = data.copy(deep=True)
    D_a_prime = data.copy(deep=True)

    # Set A = 1 for all data points in D_a and A = 0 for all data points in D_a_prime
    D_a[A] = 1
    D_a_prime[A] = 0
    
    # Predicted values based on the models
    data["D_a_predict"] = regression_model.predict(D_a)
    data["D_a_prime_predict"] = regression_model.predict(D_a_prime)
    data["propensity_score"] = propensity_model.predict(data)
    data["Y_predict"] = regression_model.predict(data)
    
    # Trimming the data
    if trim:
        data = data.query('propensity_score >= 0.1 and propensity_score <= 0.9')

    # Calculating the AIPW
    ipw = ((data[A] * data[Y]) / data["propensity_score"]) - (((1 - data[A]) * data[Y]) / (1 - data["propensity_score"]))
    backdoor = data["D_a_predict"] - data["D_a_prime_predict"]
    augmented = ((data[A] * data["Y_predict"]) / data["propensity_score"]) - (((1 - data[A]) * data["Y_predict"]) / (1 - data["propensity_score"]))
    ACE = np.mean(ipw + backdoor - augmented)
    return ACE


def compute_confidence_intervals(Y, A, Z, data, method_name, trim, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for IPW or AIPW (potentially with trimming) via bootstrap.
    The input method_name can be used to decide how to compute the confidence intervals.

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    n = data.shape[0]

    # Generate datasets by resampling with replacement and calculate estimates
    for i in range(num_bootstraps):
        estimate = method_name(Y, A, Z, data.sample(n = n, replace=True), trim)
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

    np.random.seed(100)

    nsw_observational = pd.read_csv("nsw_observational.txt")
    Y = "re78"
    A = "treat"
    # define some backdoor sets
    Z = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

    # point estimate and CIs using observational data and IPW
    print("ACE using IPW", ipw(Y, A, Z, nsw_observational, False), compute_confidence_intervals(Y, A, Z, nsw_observational, ipw, False))

    # point estimate and CIs using observational data and IPW with trimming
    print("ACE using IPW with trimming", ipw(Y, A, Z, nsw_observational, True), compute_confidence_intervals(Y, A, Z, nsw_observational, ipw, True))

    # point estimate and CIs using observational data and AIPW
    print("ACE using AIPW with trimming", augmented_ipw(Y, A, Z, nsw_observational, True), compute_confidence_intervals(Y, A, Z, nsw_observational, augmented_ipw, True))
    print("ACE using AIPW without trimming", augmented_ipw(Y, A, Z, nsw_observational, False), compute_confidence_intervals(Y, A, Z, nsw_observational, augmented_ipw, False))
    
if __name__ == "__main__":
    main()
