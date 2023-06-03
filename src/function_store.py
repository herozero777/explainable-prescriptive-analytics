import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing

from src.transition_system import transition_system, indexs_for_window, list_to_str

import pandas as pd
import pickle
import os
import random
from math import ceil
from wrapt_timeout_decorator import timeout
import warnings

# from multiprocessing import Process
# from timeout_decorator import timeout, TimeoutError
# import signal
from typing import Tuple, Any, List, Union
from collections import Counter
import utils
from time import sleep


class StoreTestRun:
    def __init__(self, save_load_path=None):
        """
        Args:
            save_load_path: E.g. "experiment_results/name_of_file.csv"
        """
        self.configs = None
        self.run_state = {
            "cfe_before_validation": [],
            "cfe_after_validation": [],
            "cfe_not_found": [],
            "cases_includes_new_data": [],
            "cases_too_small": [],
            "cases_zero_in_y": [],
            "exceptions": [],
            "cases_done": 0
        }
        save_load_dir = save_load_path.split(".")[0]
        self.save_load_state_path = save_load_dir + ".pkl"

    def add_model_configs(self, configs=None):
        """
        Args:
            configs (dict): E.g. {"window_size": 3,
                                 "reduced_kpi_time": 90,
                                 "total_cfs": 50,                       # Number of CFs DiCE algorithm should produce
                                 "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),
                                 "output_file_path": RESULTS_FILE_PATH_N_NAME,
                                 "train_dataset_size": 31_066,                                   # 31_066
                                 "proximity_weight": 0.2,
                                 "sparsity_weight": 0.2,
                                 "diversity_weight": 5.0}
        """
        if configs is None:
            raise ValueError("Pass a Python dict to the configs parameter")

        self.configs = configs

    def get_model_configs(self):
        return self.configs

    def add_cfe_to_results(self, res_tuple: Tuple = None):
        # E.g. ("cfe_not_found", cfe) = res_tuple
        dict_key, result_value = res_tuple
        self.run_state[dict_key].append(result_value)

    def save_state(self):
        with open(self.save_load_state_path, 'wb') as file:
            self.run_state["cases_done"] += 1  # Represents a single case completion
            dictionary_store = { "run_state": self.run_state,
                                 "configs": self.configs}
            print(f"====== Start Saving the result ======")
            pickle.dump(dictionary_store, file)
        print(f"====== End Saving the result For case: {self.run_state['cases_done']} ======")
        return self.run_state["cases_done"]

    def load_state(self):
        with open(self.save_load_state_path, 'rb') as file:
            dictionary_store = pickle.load(file)
            self.run_state = dictionary_store["run_state"]
            self.configs = dictionary_store["configs"]

    def get_save_load_path(self):
        return self.save_load_state_path

    def get_run_state_df(self):
        data = {"cfe_before_validation": [len(self.run_state['cfe_before_validation'])],
                "cfe_after_validation": [len(self.run_state["cfe_after_validation"])],
                "cfe_not_found": [len(self.run_state["cfe_not_found"])],
                "cases_includes_new_data": [len(self.run_state["cases_includes_new_data"])],
                "cases_too_small": [len(self.run_state["cases_too_small"])],
                "cases_zero_in_y": [len(self.run_state["cases_zero_in_y"])],
                "exceptions": [len(self.run_state["exceptions"])],
                "cases_done": [self.run_state["cases_done"]]
                }
        df_result = pd.DataFrame(data)
        return df_result


def extract_algo_name(save_load_path=""):
    file_name = save_load_path.split("/")[1]
    algo_name = file_name.split("-")[0]
    return algo_name


class Test_extract_algo_name:
    def test_end_exclusive_true(self):
        save_load_path = "experiment_results/kdtree-05-total_time.csv"
        algo_name = extract_algo_name(save_load_path)
        assert algo_name == "kdtree"

    # To test this function run command from project directory:
    # `pytest src/function_store.py `


if __name__ == '__main__':
    # Test save and load functionality
    RESULTS_FILE_PATH_N_NAME = "experiment_results/kdtree-10-total_time.csv"
    configs = {"window_size": 3,
              "reduced_kpi_time": 90,
              "total_cfs": 50,                                  # Number of CFs DiCE algorithm should produce
              "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),  # genetic, kdtree, random
              "save_load_path": RESULTS_FILE_PATH_N_NAME,
              "train_dataset_size": 31_066,                                   # 31_066
              "proximity_weight": 0.2,
              "sparsity_weight": 0.2,
              "diversity_weight": 5.0}

    state_obj = StoreTestRun(save_load_path=RESULTS_FILE_PATH_N_NAME)
    save_load_path = state_obj.get_save_load_path()

    # # Test Saving ( First run as is then comment this section and see if the save data is printed )
    # state_obj.add_model_configs(configs=configs)
    # state_obj.run_state["cfe_before_validation"].append(30)
    #
    # state_obj.save_state()

    # Test Loading
    if os.path.exists(save_load_path):
        state_obj.load_state()
        cases_done = state_obj.run_state["cases_done"]
        configs = state_obj.get_model_configs()

        print(f"configs: {configs}")
        print(f"run_state: {state_obj.run_state}")


