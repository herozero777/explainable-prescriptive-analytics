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
    def __init__(self, save_load_path=None ):
        """
        Args:
            save_load_path: E.g. "experiment_results/name_of_file.csv"
        """
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
        self.save_load_path = save_load_dir + ".pkl"

    def add_cfe_to_results(self, res_tuple: Tuple = None):
        # E.g. ("cfe_not_found", cfe) = res_tuple
        dict_key, result_value = res_tuple
        self.run_state[dict_key].append( result_value )

    def save_state(self):
        with open( self.save_load_path, 'wb' ) as file:
            self.run_state["cases_done"] += 1  # Represents a single case completion
            print(f"====== Start Saving the result ======")
            pickle.dump(self.run_state, file)
        print(f"====== End Saving the result For case: {self.run_state['cases_done']} ======")
        return self.run_state["cases_done"]

    def load_state(self):
        with open( self.save_load_path, 'rb' ) as file:
            self.run_state = pickle.load(file)

    def get_save_load_path(self):
        return self.save_load_path

    def get_run_state_df(self):
        data = {"cfe_before_validation": [ len( self.run_state['cfe_before_validation'] ) ],
                "cfe_after_validation": [ len(self.run_state["cfe_after_validation"]) ],
                "cfe_not_found": [ len(self.run_state["cfe_not_found"]) ],
                "cases_includes_new_data": [ len(self.run_state["cases_includes_new_data"])],
                "cases_too_small": [ len(self.run_state["cases_too_small"]) ],
                "cases_zero_in_y": [ len(self.run_state["cases_zero_in_y"]) ],
                "exceptions": [ len(self.run_state["exceptions"]) ],
                "cases_done": [ self.run_state["cases_done"] ]
        }
        df_result = pd.DataFrame(data)
        return df_result


