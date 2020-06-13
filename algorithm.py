import numpy as np
import sys
import warnings

from itertools import product
from matplotlib.backends import backend_gtk3

from dynamic_models.epidemic_dynamic_model import EpidemicDynamicModel
from dynamic_models.synthetic_dynamic_model_1 import SyntheticDynamicModel1
from networks.fully_connected_random_weights import FullyConnectedRandomWeights
from networks.uci_online import UCIOnline


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# Algorithm Settings
TIME_FRAMES = 100
CANDIDATE_LAMBDAS = [10 ** -i for i in range(4)]
SINDY_ITERATIONS = 10
POWER_STEP = 0.5


# Calculated Settings
POWERS = [POWER_STEP * i for i in range(1, int(2 / POWER_STEP) + 1)]
F_POWERS = POWERS
G_IJ_POWERS = list(product(POWERS, POWERS))
G_J_POWERS = POWERS
CV_PERCENT = 0.2


def _get_stacked_theta(x, adjacency_matrix):
    theta_list = []
    for node_index in range(x.shape[1]):
        x_i = x[:TIME_FRAMES, node_index]
        column_list = [
            np.ones(TIME_FRAMES),
        ]

        for f_power in F_POWERS:
            column_list.append(x_i ** f_power)

        for g_ij_power in G_IJ_POWERS:
            ij_terms = []
            for j in range(x.shape[1]):
                if j != node_index and adjacency_matrix[j, node_index]:
                    x_j = x[:TIME_FRAMES, j]
                    ij_terms.append(adjacency_matrix[j, node_index] * x_i ** g_ij_power[0] * x_j ** g_ij_power[1])
            if ij_terms:
                ij_column = np.sum(ij_terms, axis=0)
                column_list.append(ij_column)
            else:
                column_list.append(np.zeros(TIME_FRAMES))

        for g_j_power in G_J_POWERS:
            j_terms = []
            for j in range(x.shape[1]):
                if j != node_index and adjacency_matrix[j, node_index]:
                    x_j = x[:TIME_FRAMES, j]
                    j_terms.append(adjacency_matrix[j, node_index] * x_j ** g_j_power)
            if j_terms:
                j_column = np.sum(j_terms, axis=0)
                column_list.append(j_column)
            else:
                column_list.append(np.zeros(TIME_FRAMES))

        theta = np.column_stack(column_list)
        theta_list.append(theta)

    stacked_theta = np.concatenate(theta_list)
    return stacked_theta


def _sindy(x_dot, theta, candidate_lambda):
    xi = np.zeros((x_dot.shape[1], theta.shape[1]))
    for i in range(x_dot.shape[1]):
        ith_derivative = x_dot[:, i]
        ith_xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
        for j in range(SINDY_ITERATIONS):
            small_indices = np.flatnonzero(np.absolute(ith_xi) < candidate_lambda)
            big_indices = np.flatnonzero(np.absolute(ith_xi) >= candidate_lambda)
            ith_xi[small_indices] = 0
            ith_xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)[0]
        xi[i] = ith_xi
    return xi


def _optimum_sindy(x_dot, theta, x_dot_cv, theta_cv):
    least_cost = sys.maxsize
    best_xi = None
    best_mse = -1
    best_complexity = -1
    for candidate_lambda in CANDIDATE_LAMBDAS:
        xi = _sindy(x_dot, theta, candidate_lambda)
        complexity = np.count_nonzero(xi)
        x_dot_cv_hat = np.matmul(theta_cv, xi.T)
        mse = np.square(x_dot_cv - x_dot_cv_hat).mean()
        if complexity:  # zero would mean no statements
            cost = mse * complexity
            if cost < least_cost:
                least_cost = cost
                best_xi = xi
                best_mse = mse
                best_complexity = complexity
    print('best MSE:', best_mse)
    print('best complexity:', best_complexity)
    return best_xi


def _to_2d(array):
    return np.reshape(array, (array.size, 1))


def run(network_name, dynamic_model_name):
    network = None
    if network_name == FullyConnectedRandomWeights.name:
        network = FullyConnectedRandomWeights()
    elif network_name == UCIOnline.name:
        network = UCIOnline()
    else:
        print('Invalid network name')
        exit(0)

    dynamic_model = None
    if dynamic_model_name == SyntheticDynamicModel1.name:
        dynamic_model = SyntheticDynamicModel1(network)
    elif dynamic_model_name == EpidemicDynamicModel.name:
        dynamic_model = EpidemicDynamicModel(network)
    else:
        print('Invalid dynamic model name')
        exit(0)

    x = dynamic_model.get_x(TIME_FRAMES)
    stacked_theta = _get_stacked_theta(x, network.adjacency_matrix)
    y = dynamic_model.get_x_dot(x)
    stacked_x_dot = _to_2d(np.concatenate([y[:, node_index] for node_index in range(y.shape[1])]))

    cv_index = int(stacked_theta.shape[0] * (1 - CV_PERCENT))
    theta = stacked_theta[:cv_index]
    x_dot = stacked_x_dot[:cv_index]
    theta_cv = stacked_theta[cv_index:]
    x_dot_cv = stacked_x_dot[cv_index:]

    xi = _optimum_sindy(x_dot, theta, x_dot_cv, theta_cv)
    print(xi)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Invalid number of arguments')
        exit(0)
    run(sys.argv[1], sys.argv[2])
