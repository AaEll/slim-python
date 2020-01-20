from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from ortools.linear_solver import pywraplp
from copy import deepcopy
from .create_slim_IP import create_slim_IP
from .helper_functions import *
from .SLIMCoefficientConstraints import SLIMCoefficientConstraints



class SLIM(BaseEstimator, ClassifierMixin):
    """
    SLIM (Super-Sparse Linear Integer Model) is a tool for creating simple scoring systems.
    It was introduced in (Ustun 2015) Supersparse linear integer models for optimized medical scoring systems

    Estimating the model parameters is formulated as a mixted integer programming problem, with a non-convex objective.
    While objective does not guarantee reaching an optimal solution, it is formulated to induce "super" sparsity.
    It is recommended to view the model's output and re-run until a reasonable model is reached.
    """

    def __init__(self, X_names=None,hyper_params = {}):

        if X_names is not None:
            assert len(list(set(X_names))) == len(X_names), 'X_names is not unique'

        self.hyper_params ={'X_names' : X_names,
                            'C_0': 0.01,
                            'w_pos': 1.0,
                            'w_neg': 1.0,
                            'L0_min': 0,
                            'L0_max': float('inf'),
                            'err_min': 0,
                            'err_max': 1.0,
                            'pos_err_min': 0,
                            'pos_err_max': 1.0,
                            'neg_err_min': 0,
                            'neg_err_max': 1.0,
                            'timelimit': 300
                            }

        self.hyper_params.update(hyper_params)
        self.str_representation = None


    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        N, P = X.shape

        hyper_params = deepcopy(self.hyper_params)

        if hyper_params['X_names'] is None:
            hyper_params['X_names'] = ['feature_'+str(j) for j in range(P)]

        if len(y.shape) == 1:
            y = y[:,None] # add dimension to y

        self.__fit_encoder__(X, 2)

        self.X_, hyper_params['X_names'] = self.__transform_encoder__(X,hyper_params['X_names'])
        self.y_ = y

        _, P = self.X_.shape
        assert P == len(hyper_params['X_names'])

        hyper_params['coef_constraints'] = SLIMCoefficientConstraints(variable_names = hyper_params['X_names'], XY = self.X_*self.y_, ub = 5, lb = -5)

        assert all((self.y_ == 1)|(self.y_ == -1)) or all((self.y_ == 1)|(self.y_ == 0)), 'Y[i] should = [-1,1] or [0,1] for all i'
        assert N > 0, 'X matrix must have at least 1 row'
        assert P > 0, 'X matrix must have at least 1 column'
        assert len(self.y_) == N, 'len(Y) should be same as # of rows in X'
        assert len(hyper_params['X_names'] ) == P, 'len(X_names) should be same as # of cols in X'
        assert all(((self.X_ == 1) | (self.X_ == 0)).flatten()), 'X[i,j] should = [0,1] for all i,j'

        # replace 0 with -1, so Y[i] should = [-1,1] for all i
        self.y_ = np.where(y == 0, -1, y)

        slim_solver, slim_info = create_slim_IP(self.X_,self.y_,hyper_params)
        #TODO set hyper_params['timelimit'] as max time for solver

        # NOTE this will crash the kernal if constraint names are not unique
        status = slim_solver.Solve()

        rho_values = np.array([slim_info['variables'][rho_name].solution_value()
                      for rho_name in slim_info['rho_names']])

        if status == pywraplp.Solver.OPTIMAL:
            status_code = 0
        elif status == pywraplp.Solver.FEASIBLE:
            status_code = 0
        else:
            status_code = 1


        #MIP Related Items
        self.slim_summary_ = {
            #
            # IP related information
            #
            'solution_status_code': status_code,
            'solution_status': status,
            'objective_value': slim_solver.Objective().Value(),
            'simplex_iterations': slim_solver.Iterations(),
            'nodes_processed': slim_solver.nodes()
        }
        """ # populated in the line below
            # Solution based information (default values)
            #
            'rho': np.nan,
            'pretty_model': np.nan,
            'string_model': np.nan,
            'true_positives': np.nan,
            'true_negatives': np.nan,
            'false_positives': np.nan,
            'false_negatives': np.nan,
            'mistakes': np.nan,
            'error_rate': np.nan,
            'true_positive_rate': np.nan,
            'false_positive_rate': np.nan,
            'L0_norm': np.nan,
        }
        """
        self.slim_summary_.update(get_rho_summary(rho_values, slim_info, self.X_, self.y_
        ))

        #print(slim_summary)

        # print metrics from slim_summary
        print('simplex_iterations: ' + str(self.slim_summary_['simplex_iterations']))
        print('solution_status: ' + str(self.slim_summary_['solution_status']))

        # print model
        print("Model")
        print(self.slim_summary_['string_model'])

        # print coefficient vector
        print("Coefficient Vector: " + str(self.slim_summary_['rho']))

        # print accuracy metrics
        print('error_rate: {:.2f}'.format(100*self.slim_summary_['error_rate']))
        print('TPR: {:.2f}'.format(100*self.slim_summary_['true_positive_rate']))
        print('FPR: {:.2f}'.format(100*self.slim_summary_['false_positive_rate']))
        print('true_positives: {:d}'.format(self.slim_summary_['true_positives']))
        print('false_positives: {:d}'.format(self.slim_summary_['false_positives']))
        print('true_negatives: {:d}'.format(self.slim_summary_['true_negatives']))
        print('false_negatives: {:d}'.format(self.slim_summary_['false_negatives']))

        # Return the classifier
        return self


    def __fit_encoder__(self,X,nclasses):
        #transformer1 = KBinsDiscretizer(n_bins = nclasses, encode = 'ordinal')
        #transformer2 = ThermoEncoder()
        #pipeline = Pipeline([('discretize',transformer1),('encoder',transformer2)])

        #for col in zip(nclasses, X.T):
        #    if all((col == 1) | (col == 0)): #column is binary
        #        continue
        #    else: # otherwise it needs to be transformed
        #self.encoder_.fit(X)
        return X


    def __transform_encoder__(self,X, X_names = None):

        #check_is_fitted(self,['encoder_'])

        X_transformed = X # self.encoder_.transform(X)

        if '__Intercept__' not in X_names:
            X_transformed = np.insert(arr = X_transformed, obj = 0, values = np.ones(X.shape[0]), axis = 1)

            if X_names is not None:
                X_names.insert(0, '__Intercept__')

        X_names_transformed = X_names# TODO

        return X_transformed, X_names_transformed




    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self,['slim_summary_'])

        # Input validation
        X = check_array(X)

        X,_ = __transform_encoder__(X)

        yhat = X.dot(self.slim_summary_['rho'][:,None]) > 0
        yhat = np.array(yhat, dtype = np.float)

        return yhat

    def __str__(self):

        check_is_fitted(self,['slim_summary_'])

        if self.slim_summary_ is not None and "string_model" in self.slim_summary_:
            return self.slim_summary_["string_model"]
        else:
            return "SLIM Model : Uninitialized"
