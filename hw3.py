#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np


def ipw(Y, A, Z, data, trim=False):
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

    # code for IPW
    return 0

def augmented_ipw(Y, A, Z, data, trim=False):
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

    # code for AIPW
    return 0

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

    # point estimate and CIs using observational data and IPW with trimming=
    print("ACE using IPW with trimming", 0, (-1, 1))

    # point estimate and CIs using observational data and AIPW
    print("ACE using AIPW (potentially with trimming)", (0, -1, 1))


if __name__ == "__main__":
    main()
