import os
import sys
import numpy as np
import pandas as pd
import cplex as cp
import createIP
import helperFunctions as helper
from coeffConstraints import SLIMCoefficientConstraints

#### LOAD DATA ####
# requirements for CSV data file
# - outcome variable in first column
# - outcome variable values should be [-1, 1] or [0, 1]
# - first row contains names for the outcome variable + input variables
# - no empty cells

# Call: python slim.py <file path> <# seconds>
if __name__ == "__main__":

    data_csv_file = sys.argv[1] # full path to CSV file
    runTime = float(sys.argv[2])  # time in seconds

    # load data file from csv
    df = pd.read_csv(data_csv_file, sep = ',')
    data = df.as_matrix()
    data_headers = list(df.columns.values)
    N = data.shape[0]

    # setup Y vector and Y_name
    Y_col_idx = [0]
    Y = data[:, Y_col_idx]
    Y_name = [data_headers[j] for j in Y_col_idx]
    Y[Y == 0] = -1 # {0,1} outcome data converted to {-1,1}

    # setup X and X_names
    X_col_idx = [j for j in range(data.shape[1]) if j not in Y_col_idx]
    X = data[:, X_col_idx]
    X_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    X = np.insert(arr = X, obj = 0, values = np.ones(N), axis = 1)
    X_names.insert(0, '(Intercept)')

    # run sanity checks
    helper.check_data(X = X, Y = Y, X_names = X_names)

    #### TRAIN SCORING SYSTEM USING SLIM ####
    # setup SLIM coefficient set
    coef_constraints = SLIMCoefficientConstraints(variable_names = X_names, ub = 5, lb = -5)

    #choose upper and lower bounds for the intercept coefficient
    #to ensure that there will be no regularization due to the intercept, choose
    #
    #intercept_ub < min_i(min_score_i)
    #intercept_lb > max_i(max_score_i)
    #
    # where min_score_i = min((Y*X) * \rho) for rho in \Lset
    # where max_score_i = max((Y*X) * \rho) for rho in \Lset
    #
    # setting intercept_ub and intercept_lb in this way ensures that we can always
    # classify every point as positive and negative

    # Note: for np.ndarray, * is element by element multiplication

    scores_at_ub = (Y * X) * coef_constraints.ub
    scores_at_lb = (Y * X) * coef_constraints.lb

    # exclude intercept scores
    non_intercept_ind = np.array([n != '(Intercept)' for n in X_names]) # names of all non-intercept variables
    scores_at_ub = scores_at_ub[:, non_intercept_ind]
    scores_at_lb = scores_at_lb[:, non_intercept_ind]

    max_scores = np.fmax(scores_at_ub, scores_at_lb) # abs max value (among lb and ub score) for each location in matrix
    min_scores = np.fmin(scores_at_ub, scores_at_lb)
    max_tot_scores = np.sum(max_scores, 1) # sums scores for each variable horizontally
    min_tot_scores = np.sum(min_scores, 1)

    intercept_ub = -min(min_tot_scores) + 1
    intercept_lb = -max(max_tot_scores) + 1

    coef_constraints.set_field('ub', '(Intercept)', intercept_ub)
    coef_constraints.set_field('lb', '(Intercept)', intercept_lb)
    print(coef_constraints)

    #create SLIM IP
    slim_input = {
        'X': X,
        'X_names': X_names,
        'Y': Y,
        'C_0': 0.01,
        'w_pos': 2.0,
        'w_neg': 1.0,
        'L0_min': 0,
        'L0_max': float('inf'),
        'err_min': 0,
        'err_max': 1.0,
        'pos_err_min': 0,
        'pos_err_max': 1.0,
        'neg_err_min': 0,
        'neg_err_max': 1.0,
        'coef_constraints': coef_constraints
    }

    slim_IP, slim_info = createIP.create_slim_IP(slim_input)

    # setup SLIM IP parameters
    # see docs/usrccplex.pdf for more about these parameters
    slim_IP.parameters.timelimit.set(runTime) #set runtime here

    #TODO: add these default settings to create_slim_IP
    slim_IP.parameters.randomseed.set(0)
    slim_IP.parameters.threads.set(1)
    slim_IP.parameters.parallel.set(1)
    slim_IP.parameters.output.clonelog.set(0)
    slim_IP.parameters.mip.tolerances.mipgap.set(np.finfo(np.float).eps)
    slim_IP.parameters.mip.tolerances.absmipgap.set(np.finfo(np.float).eps)
    slim_IP.parameters.mip.tolerances.integrality.set(np.finfo(np.float).eps)
    slim_IP.parameters.emphasis.mip.set(1)


    # solve SLIM IP
    slim_IP.solve()

    # run quick and dirty tests to make sure that IP output is correct
    helper.check_slim_IP_output(slim_IP, slim_info, X, Y, coef_constraints)

    #### CHECK RESULTS ####
    slim_results = helper.get_slim_summary(slim_IP, slim_info, X, Y)
    #print(slim_results)

    # print metrics from slim_results
    print('simplex_iterations: ' + str(slim_results['simplex_iterations']))
    print('solution_status: ' + str(slim_results['solution_status']))
    print('objval_lowerbound: ' + str(slim_results['objval_lowerbound']))
    print('objective_value: ' + str(slim_results['objective_value']))
    print('optimality_gap: ' + str(slim_results['optimality_gap']))

    # print model
    print("Model")
    print(slim_results['string_model'])

    # print coefficient vector
    print("Coefficient Vector: " + str(slim_results['rho']))

    # print accuracy metrics
    print('error_rate: {:.2f}'.format(100*slim_results['error_rate']))
    print('TPR: {:.2f}'.format(100*slim_results['true_positive_rate']))
    print('FPR: {:.2f}'.format(100*slim_results['false_positive_rate']))
    print('true_positives: {}'.format(slim_results['true_positives']))
    print('false_positives: {}'.format(slim_results['false_positives']))
    print('true_negatives: {}'.format(slim_results['true_negatives']))
    print('false_negatives: {}'.format(slim_results['false_negatives']))
