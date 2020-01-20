from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from .create_slim_IP import create_slim_IP

class SLIM(BaseEstimator, ClassifierMixin):
    """
    SLIM (Super-Sparse Linear Integer Model) is a tool for creating simple scoring systems.
    It was introduced in (Ustun 2015) Supersparse linear integer models for optimized medical scoring systems

    Estimating the model parameters is formulated as a mixted integer programming problem, with a non-convex objective.
    While objective does not guarantee reaching an optimal solution, it is formulated to induce "super" sparsity.
    It is recommended to view the model's output and re-run until a reasonable model is reached.
    """

    def __init__(self, hyper_param=None):
        if hyper_param is None:
            #default param
            hyper_param = {}
        self.hyper_param = hyper_param
        self.str_representation = None


    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert self.classes_ == 2
        assert all((Y == 1)|(Y == -1)) or all((Y == 1)|(Y == 0)), 'Y[i] should = [-1,1] or [0,1] for all i'

        # replace 0 with -1, so Y[i] should = [-1,1] for all i
        y = np.where(y == 0, -1, y)


        slim_solver, slim_info = create_slim_IP.create_slim_IP(X,y,self.hyper_param)

        slim_solver.parameters.max_time_in_seconds = self.hyper_param['timelimit']

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
        self.slim_summary = {
            #
            # IP related information
            #
            'solution_status_code': status_code,
            'solution_status': status,
            'objective_value': slim_solver.Objective.Value(),
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
        self.slim_summary.update(get_rho_summary(rho_values, slim_info, X, Y))

        #print(slim_summary)

        # print metrics from slim_summary
        print('simplex_iterations: ' + str(slim_summary['simplex_iterations']))
        print('solution_status: ' + str(slim_summary['solution_status']))

        # print model
        print("Model")
        print(slim_summary['string_model'])

        # print coefficient vector
        print("Coefficient Vector: " + str(slim_summary['rho']))

        # print accuracy metrics
        print('error_rate: {:.2f}'.format(100*slim_summary['error_rate']))
        print('TPR: {:.2f}'.format(100*slim_summary['true_positive_rate']))
        print('FPR: {:.2f}'.format(100*slim_summary['false_positive_rate']))
        print('true_positives: {:d}'.format(slim_summary['true_positives']))
        print('false_positives: {:d}'.format(slim_summary['false_positives']))
        print('true_negatives: {:d}'.format(slim_summary['true_negatives']))
        print('false_negatives: {:d}'.format(slim_summary['false_negatives']))

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        yhat = X.dot(self.slim_summary['rho']) > 0
        yhat = np.array(yhat, dtype = np.float)

        return yhat

    def __str__(self):

        check_is_fitted(self)

        if self.slim_summary is not None and "string_model" in self.slim_summary:
            return self.slim_summary["string_model"]
        else:
            return "SLIM Model : Uninitialized"
