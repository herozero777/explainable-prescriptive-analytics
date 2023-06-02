import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from src.transition_system import transition_system, indexs_for_window, list_to_str

from datetime import datetime
import pandas as pd
import pickle
import os
from time import time
import random
from math import ceil
from wrapt_timeout_decorator import timeout
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
################################
# Helper Functions
################################


class StoreTestRun:
    def __init__(self, save_load_path= None ):
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

    def add_cfe_to_results(self, res_tuple = None):
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


def get_case_id(df, case_id_name="SR_Number"):  # , multi=False
    return df[case_id_name].unique().item()


def get_query_instance(sidx=14, eidx=16):
    assert eidx - sidx == 2, "One row represents the current action and the next one represents the suggested action"
    current_step = X_train[sidx: sidx+1]
    expected_next_step = X_train[eidx-1: eidx]
    return current_step, expected_next_step


def activity_n_resources(df, resources_columns=None, threshold_percentage=100):
    """
    Creates a set of tuples, each tuple has elements from the columns specified through `resource_columns`.
    E.g. { ('action_1', 'rresource_1'), ... (activity_3, resource_5) }
    Args:
        df (pd.DataFrame):
        resources_columns (list): columns that contains the activity and resources.
    Returns:
        Set of tuples. A single element contains the activity and resources of a single row from the
        dataframe.
    """
    if resources_columns is None:
        # raise TypeError("Please specify the columns that have resources")
        resources_columns = [activity_column_name, 'Involved_ST_Function_Div', 'Involved_Org_line_3',
                             'Involved_ST', 'Country', 'Owner_Country']

    threshold = threshold_percentage / 100

    valid_activity_n_resource = set( df[resources_columns].apply(tuple, axis='columns') )

    # combo: combination
    resource_combo_frequency = {}

    valid_resource_combo = df[resources_columns].apply(tuple, axis='columns')

    for elem in valid_resource_combo:
        if resource_combo_frequency.get(elem):
            resource_combo_frequency[elem] += 1
        else:
            resource_combo_frequency[elem] = 1
    # Creates a list of (combo, counts)
    resource_combo_counts = [ (k, v) for k, v in resource_combo_frequency.items() ]
    sorted_resource_combo_counts = sorted( resource_combo_counts, key=lambda item: item[1], reverse=True )
    sorted_combos = [combo for combo, _ in sorted_resource_combo_counts ]
    amount_of_combos_to_select = int( len(sorted_combos) * threshold ) + 1
    valid_activity_n_resources = sorted_combos[:amount_of_combos_to_select]
    return valid_activity_n_resources


def get_test(df, case_id_name):
    # df_result = pd.DataFrame(columns=df.columns)
    result_lst = []

    for idx in df[case_id_name].unique():
        df_trace = df[df[case_id_name] == idx]
        # ceil enables cases with 1 row to pass through
        cut = ceil(len(df_trace) * random.uniform(0.5, 0.7)) #+ 2  # 2 because one for the floor and one for the pred
        df_trace = df_trace.iloc[:cut].reset_index(drop=True)

        # df_result = pd.concat([df_result, df_trace])
        result_lst.append(df_trace.reset_index(drop=True))
        # break
    # return df_result.reset_index(drop=True)
    return result_lst


def get_prefix_of_activities(expected_activity_index=None, event_log=None, df_single_trace=None, window_size=3, activity_column_name=None):
    """ Retrieves the prefix of activities from the event log. So that later the next activity can be validated using the prefix.
    This function can be used for 2 different cases. 1st, passing different arguments allowing is to single out trace prefix
    from the entire event log (df_train). 2nd, passing it df_single_trace and prefix is extracted from it.
    Args:
        expected_activity_index (int)
        event_log (pd.DataFrame): Dataframe containing many traces. E.g. df_train
        df_single_trace (pd.DataFrame): A dataframe that contains a single trace. E.g. a running trace or a test trace. It is expected
            that the index of this dataframe starts from 0. An assumption is that last activity/ row represents the expected activity,
            so the prefix of activities ends at the 2nd last activity. when using this parameter, query_case_id and related parameters
            are ignored.

    """
    # Error checking
    if activity_column_name is None:
        raise "Please specify activity_column_name"

    if df_single_trace is not None:  # Case 1
        # Check if indexes start from 0
        assert df_single_trace.loc[0] is not None

        # Due to assumption that last activity is the expected activity so the prefix ends at the 2nd last activity
        index_to_previous_activity = df_single_trace.index[-2]

        start_index, end_index = indexs_for_window(index_to_previous_activity, window_size=window_size, end_exclusive=False)
        prefix_of_activities = df_single_trace.loc[start_index: end_index, activity_column_name].to_list()  # loc is used to access the index values inside the dataframe
        prefix_of_activities = list_to_str(prefix_of_activities)

        return prefix_of_activities
    else:  # Case 2
        # if query_case_id is None:
        #     raise "Please specify query_case_id!"
        if event_log is None:
            raise "Please specify event_log!"
        if expected_activity_index is None:
            raise "Please specify expected_activity_index!"

        query_case_id = get_case_id( event_log[expected_activity_index: expected_activity_index+1], case_id_name=case_id_name )

        # Isolate the query_case_id trace
        df_query = event_log[ event_log[case_id_name] == query_case_id ]

        # Prefix ends before the expected activity timestamp
        index_to_previous_activity = expected_activity_index - 1

        start_index, end_index = indexs_for_window(index_to_previous_activity, window_size=window_size, end_exclusive=False)
        prefix_of_activities = df_query.loc[start_index: end_index, activity_column_name].to_list()  # loc is used to access the index values inside the dataframe
        prefix_of_activities = list_to_str(prefix_of_activities)

        return prefix_of_activities


def validate_transition(cfe, prefix_of_activities=None, transition_graph=None, valid_resources=None):
    """  resource_columns_to_validate=None possible future parameter
    Args:
        cfe (dice_ml.counterfactual_explanations.CounterfactualExplanations): Dice counterfactual explanations object.
        window_size (int): Size of the prefix of trace for which next activity is checked. See `index_for_window` function
                            documentation.
        expected_activity_index (int):
    Returns:
        pd.DataFrame
    """
    if cfe is None:
        raise "Please specify cfe!"
    if valid_resources is None:
        raise "Please specify valid_resources!"
    if transition_graph is None:
        raise "Please specify transition_graph"
    if prefix_of_activities is None:
        raise "Please specify prefix_of_activities"

    cf_examples_df = cfe.cf_examples_list[0].final_cfs_df.copy()  # Get the counterfactual explanations dataframe from the object

    # === Verify the next activity
    indexes_to_drop = []
    for i, suggested_next_activity in cf_examples_df[activity_column_name].items():
        if suggested_next_activity not in transition_graph[prefix_of_activities]:
            indexes_to_drop.append(i)
            # print(i, suggested_next_activity)

    cf_examples_df = cf_examples_df.drop(indexes_to_drop, axis='index').reset_index(drop=True)

    # === Verify the associated resources
    indexes_to_drop = []
    for i, row in cf_examples_df[ resource_columns_to_validate ].iterrows():
        row_tuple = tuple(row)
        if row_tuple not in valid_resources:
            # print(f"removed row had: {row_tuple}")
            indexes_to_drop.append(i)

    cf_examples_df = cf_examples_df.drop(indexes_to_drop, axis='index').reset_index(drop=True)
    return cf_examples_df


@timeout(120)  # Timeout unit seconds
def generate_cfe(query_instances, total_time_upper_bound=None, total_cfs=50, KPI="activity_occurrence"):
    """
    Args:
        query_instances (pd.DataFrame):
        total_time_upper_bound (int): The upper value of the target (y) label.
        total_cfs (int): Number of Counterfactual examples (CFEs) to produce via `generate_counterfactuals()`

    Returns:
        cfe (dice_ml.counterfactual_explanations.CounterfactualExplanations): Dice counterfactual explanations object.
    """
    if KPI == "activity_occurrence":
        cfe = exp_genetic_iris.generate_counterfactuals(query_instances, total_CFs=15, desired_class="opposite", features_to_vary=cols_to_vary,
                                                        permitted_range = {"ACTIVITY": ['Service closure Request with network responsibility',
                                                                                'Service closure Request with BO responsibility',
                                                                                'Pending Request for Reservation Closure', 'Pending Liquidation Request',
                                                                                'Request created','Authorization Requested', 'Evaluating Request (NO registered letter)',
                                                                                'Network Adjustment Requested', 'Evaluating Request (WITH registered letter)',
                                                                                'Pending Request for Network Information']})  # 'Back-Office Adjustment Requested'
    else:
        cfe = exp_genetic_iris.generate_counterfactuals(query_instances, total_CFs=total_cfs, desired_range=[0, total_time_upper_bound], features_to_vary=cols_to_vary,
                                                    proximity_weight=proximity_weight, sparsity_weight=sparsity_weight, diversity_weight=diversity_weight)
    return cfe


if __name__ == '__main__':
    start_time = time()
    print(f"========================= Program Start at: {datetime.fromtimestamp(start_time)} =========================")
    # Get the path of the current script file
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Print the directory path
    print("Current Working Directory:", os.getcwd())
    print(script_path)

    KPI = "activity_occurrence"
    SECONDS_TO_HOURS = 60 * 60
    SECONDS_TO_DAYS = 60 * 60 * 24
    WINDOW_SIZE = 3
    TOTAL_CFS = 500                        # Number of CFs DiCE algorithm should produce
    TRAIN_DATA_SIZE = 170_335              # 170_335
    DICE_METHOD = "genetic"
    RESULTS_FILE_PATH_N_NAME = "experiment_results/genetic-01-activity-occu.csv"
    proximity_weight = 0.2  # 0.2
    sparsity_weight = 0.2  # 0.2
    diversity_weight = 5.0  # 5.0

    parameter_configs = {"Window_size": WINDOW_SIZE, "Total CFS": TOTAL_CFS,
                         "DiCE Algo Method": DICE_METHOD, "output_file_path": RESULTS_FILE_PATH_N_NAME,
                         "Traning Dataset Size": TRAIN_DATA_SIZE,
                         "proximity_weight": proximity_weight,
                         "sparsity_weight": sparsity_weight,
                         "diversity_weight": diversity_weight}
    print("Configs:", parameter_configs)

    case_id_name = 'REQUEST_ID'  # The case identifier column name.
    activity_column_name = "ACTIVITY"


    data_dir = "./preprocessed_datasets/"
    train_dataset_file = "bank_acc_train.csv"
    # test_dataset_file = "bank_acc_test.csv"
    test_pickle_dataset_file = "bank_acc-test.pkl"
    df = pd.read_csv("./data/completed.csv")  # Use full dataset for transition systens
    df_train = pd.read_csv(os.path.join(data_dir, train_dataset_file))
    # df_test = pd.read_csv(os.path.join(data_dir, test_dataset_file))

    # Some Basic Preprocessing
    df = df.fillna("missing")

    # # Temporary
    df_train = df_train[:TRAIN_DATA_SIZE]
    ## ---------

    resource_columns_to_validate = [activity_column_name, 'CE_UO', 'ROLE']
    valid_resources = activity_n_resources(df, resource_columns_to_validate)
    # len(valid_resources)

    # test_cases = get_test(df_test, case_id_name)
    # Unpickle the Standard test-set. To standardize the test across different parameters.
    with open(os.path.join(data_dir, test_pickle_dataset_file), 'rb') as file:
        test_cases = pickle.load(file)

    cols_to_vary = ["ACTIVITY", "CE_UO", "ROLE"]

    outcome_name = "Back-Office Adjustment Requested"

    def prepare_df_for_ml(df, outcome_name, columns_to_remove=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param str outcome_name: name of the target column.
        """
        # Before training for ml we need to remove columns that can are not needed for ML model.
        if columns_to_remove is None:
            columns_to_remove = ["Change_Date+Time", "time_remaining"]
        df = df.drop([case_id_name], axis="columns")
        df = df.drop(columns_to_remove, axis="columns")
        X = df.drop([outcome_name], axis=1)
        y = df[outcome_name]
        return X, y

    X_train, y_train = prepare_df_for_ml(df_train, outcome_name, columns_to_remove=["START_DATE", "END_DATE", "time_remaining"])

    continuous_features = ["time_from_first", "time_from_previous_et", "time_from_midnight", "activity_duration",
                           '# ACTIVITY=Service closure Request with network responsibility',
                           '# ACTIVITY=Service closure Request with BO responsibility',
                           '# ACTIVITY=Pending Request for Reservation Closure',
                           '# ACTIVITY=Pending Liquidation Request',
                           '# ACTIVITY=Request completed with account closure', '# ACTIVITY=Request created',
                           '# ACTIVITY=Authorization Requested',
                           '# ACTIVITY=Evaluating Request (NO registered letter)',
                           '# ACTIVITY=Network Adjustment Requested',
                           '# ACTIVITY=Pending Request for acquittance of heirs',
                           '# ACTIVITY=Request deleted', '# ACTIVITY=Back-Office Adjustment Requested',
                           '# ACTIVITY=Evaluating Request (WITH registered letter)',
                           '# ACTIVITY=Request completed with customer recovery',
                           '# ACTIVITY=Pending Request for Network Information', ]
    categorical_features = ["CLOSURE_TYPE", "CLOSURE_REASON", "ACTIVITY", "CE_UO", "ROLE", "weekday"]

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier(n_jobs=7))])
    model = clf.fit(X_train, y_train)

    print("=================== Create DiCE model ===================")
    # ## Create DiCE model
    d_iris = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis="columns"),
                          continuous_features=continuous_features,
                          outcome_name=outcome_name)

    # We provide the type of model as a parameter (model_type)
    m_iris = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
    method = DICE_METHOD  # genetic, kdtree, random
    # Categorical features do not support features_weights argument in generate_counterfactuals()
    exp_genetic_iris = Dice(d_iris, m_iris, method=method)

    # === Load the Transition Graph
    _, transition_graph = transition_system(df, case_id_name=case_id_name, activity_column_name=activity_column_name,
                                            window_size=WINDOW_SIZE)

    state_obj = StoreTestRun(save_load_path=RESULTS_FILE_PATH_N_NAME)
    save_load_path = state_obj.get_save_load_path()

    if os.path.exists(save_load_path):
        state_obj.load_state()
        cases_done = state_obj.run_state["cases_done"]
    else:
        cases_done = 0

    print("=================== Create CFEs for all the test cases ===================")

    for df_test_trace in test_cases[cases_done:]:

        query_case_id = get_case_id(df_test_trace, case_id_name=case_id_name)

        if 0 < len(df_test_trace) <= 2:
            print("too small", cases_done, df_test_trace[case_id_name].unique().item())
            result_value = query_case_id
            state_obj.add_cfe_to_results(("cases_too_small", result_value))
            cases_stored = state_obj.save_state()
            cases_done += 1
            continue

        X_test, y_test = prepare_df_for_ml(df_test_trace, outcome_name, columns_to_remove=["START_DATE", "END_DATE", "time_remaining"])

        # Check if y_test is 0 then don't generate CFE
        if y_test.iloc[-1] == 0:
            result_value = query_case_id
            state_obj.add_cfe_to_results(("cases_zero_in_y", result_value))
            cases_stored = state_obj.save_state()
            cases_done += 1
            continue

        # Access the last row of the truncated trace to replicate the behavior of a running trace
        query_instances = X_test.iloc[-1:]
        try:
            cfe = generate_cfe(query_instances, total_cfs=TOTAL_CFS, KPI="activity_occurrence")
            result_value = (query_case_id, cfe)
            state_obj.add_cfe_to_results(("cfe_before_validation", result_value))  # save after cfe validation

            prefix_of_activities = get_prefix_of_activities(df_single_trace=df_test_trace, window_size=0, activity_column_name=activity_column_name)
            cfe_df = validate_transition(cfe, prefix_of_activities=prefix_of_activities, transition_graph=transition_graph, valid_resources=valid_resources)

            if len(cfe_df) > 0:
                result_value = (query_case_id, cfe_df)
                state_obj.add_cfe_to_results(("cfe_after_validation", result_value))

            cases_stored = state_obj.save_state()

        except UserConfigValidationException:
            result_value = query_case_id
            state_obj.add_cfe_to_results(("cfe_not_found", result_value))
            cases_stored = state_obj.save_state()
        except TimeoutError as err:  # When function takes too long
            result_value = query_case_id
            print("TimeoutError caught:", err)
            state_obj.add_cfe_to_results(("cfe_not_found", result_value))
            cases_stored = state_obj.save_state()
        except ValueError:
            # print(f"Includes feature not found in training data: {get_case_id(df_test_trace)}")
            result_value = query_case_id
            state_obj.add_cfe_to_results(("cases_includes_new_data", result_value))
            cases_stored = state_obj.save_state()
        # This error is seen occurring on when running lots of loops on the server
        except AttributeError as e:
            print("AttributeError caught:", e)
            state_obj.add_cfe_to_results(("exceptions", query_case_id))
            cases_stored = state_obj.save_state()
        except Exception as err:
            print(f"Broadest Exception handler invoked", err)
            state_obj.add_cfe_to_results(("exceptions", query_case_id))
            cases_stored = state_obj.save_state()

        # For printing results progressively
        if (cases_done % 100) == 0:
            df_result = state_obj.get_run_state_df()
            df_result.to_csv(RESULTS_FILE_PATH_N_NAME, index=False)

        cases_done += 1
        # if i >= 20:
        #     break
        # ----------------------------------------------------------------

    df_result = state_obj.get_run_state_df()
    df_result.to_csv(RESULTS_FILE_PATH_N_NAME, index=False)

    print(f"Time it took: { round( ((time() - start_time) / SECONDS_TO_HOURS), 3) }")
    print("======================================== Testing Complete !!! =============================================")
