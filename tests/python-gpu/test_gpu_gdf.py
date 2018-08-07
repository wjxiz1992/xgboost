import numpy as np
import pandas as pd
import pygdf.dataframe as gdf
from sklearn import datasets
import sys
import unittest
import xgboost as xgb

from regression_test_utilities import run_suite, parameter_combinations, \
    assert_results_non_increasing, Dataset


def get_gdf():
    rng = np.random.RandomState(199)
    n = 50000
    m = 20
    sparsity = 0.25
    X, y = datasets.make_regression(n, m, random_state=rng)
    #X = np.array([[np.nan if rng.uniform(0, 1) < sparsity else x
    #               for x in x_row] for x_row in X])
    X = np.ascontiguousarray(np.transpose(X))
    df = gdf.DataFrame(list(zip(['col%d' % i for i in range(m)], X)))
    print('n_columns =', len(df.columns))
    print('n_rows =', len(df))
    return df, y
    

class TestGPU(unittest.TestCase):

    gdf_datasets = [Dataset("GDF", get_gdf, "gpu:reg:linear", "rmse")]
    
    def test_gdf(self):
        variable_param = {'n_gpus': [1], 'max_depth': [10], 'max_leaves': [255],
                          'max_bin': [255],
                          'grow_policy': ['lossguide']}
        for param in parameter_combinations(variable_param):
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, num_rounds=20,
                                    select_datasets=self.gdf_datasets)
            print(gpu_results)
            assert_results_non_increasing(gpu_results, 1e-2)
            #param['tree_method'] = 'hist'
            #cpu_results = run_suite(param, select_datasets=self.gdf_datasets)
            #assert_gpu_results(cpu_results, gpu_results)
