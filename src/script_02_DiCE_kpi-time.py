import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from src.transition_system import transition_system, indexs_for_window, list_to_str

from datetime import datetime
import pandas as pd
import pickle
import os
from time import time
import random
from math import ceil
from wrapt_timeout_decorator import timeout


################################
# Helper Functions
################################


def get_case_id(df, case_id_name="SR_Number", multi=False):
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

        query_case_id = get_case_id( event_log[expected_activity_index: expected_activity_index+1] )

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
def generate_cfe(query_instances, total_time_upper_bound, total_cfs=50):
    """
    Args:
        query_instances (pd.DataFrame):
        total_time_upper_bound (int): The upper value of the target (y) label.
        total_cfs (int): Number of Counterfactual examples (CFEs) to produce via `generate_counterfactuals()`

    Returns:
        cfe (dice_ml.counterfactual_explanations.CounterfactualExplanations): Dice counterfactual explanations object.
    """
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

    SECONDS_TO_HOURS = 60 * 60
    SECONDS_TO_DAYS = 60 * 60 * 24
    WINDOW_SIZE = 3
    REDUCED_KPI_TIME = 90
    TOTAL_CFS = 500                        # Number of CFs DiCE algorithm should produce
    TRAIN_DATA_SIZE = 31_066               # 31_066
    DICE_METHOD = "random"
    RESULTS_FILE_PATH_N_NAME = "experiment_results/random-02-total-time.csv"
    proximity_weight = 1.0  # 0.2
    sparsity_weight = 1.0  # 0.2
    diversity_weight = 5.0  # 5.0

    parameter_configs = {"Window_size": WINDOW_SIZE, "Reducing Time by %": REDUCED_KPI_TIME, "Total CFS": TOTAL_CFS,
                         "DiCE Algo Method": DICE_METHOD, "output_file_path": RESULTS_FILE_PATH_N_NAME,
                         "Traning Dataset Size": TRAIN_DATA_SIZE,
                         "proximity_weight": proximity_weight,
                         "sparsity_weight": sparsity_weight,
                         "diversity_weight": diversity_weight}
    print("Configs:", parameter_configs)

    case_id_name = 'SR_Number'  # The case identifier column name.
    start_date_name = 'Change_Date+Time'  # Maybe change to start_et (start even time)
    activity_column_name = "ACTIVITY"

    data_dir = "preprocessed_datasets/"
    train_dataset_file = "train-set-cfe.csv"
    # test_dataset_file = "test-set-cfe.csv"
    test_pickle_dataset_file = "test-set-cfe.pkl"
    df = pd.read_csv("data/VINST cases incidents.csv")  # Use full dataset for transition systens
    df_train = pd.read_csv(os.path.join(data_dir, train_dataset_file))
    # df_test = pd.read_csv(os.path.join(data_dir, test_dataset_file))

    # Some Basic Preprocessing
    df = df.fillna("missing")

    # # Temporary
    df_train = df_train[:TRAIN_DATA_SIZE]
    ## ---------

    resource_columns_to_validate = [activity_column_name, 'Involved_ST_Function_Div', 'Involved_Org_line_3',
                                    'Involved_ST', 'Country', 'Owner_Country']
    valid_resources = activity_n_resources(df, resource_columns_to_validate)
    # len(valid_resources)

    # test_cases = get_test(df_test, case_id_name)
    # Unpickle the Standard test-set. To standardize the test across different parameters.
    with open(os.path.join(data_dir, test_pickle_dataset_file), 'rb') as file:
        test_cases = pickle.load(file)

    cols_to_vary = ["ACTIVITY", "Involved_ST_Function_Div", "Involved_Org_line_3", "Involved_ST"]

    outcome_name = "lead_time"

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


    X_train, y_train = prepare_df_for_ml(df_train, outcome_name)

    continuous_features = ["time_from_first", "time_from_previous_et", "time_from_midnight", "# ACTIVITY=In Progress",
                           "# ACTIVITY=Awaiting Assignment",
                           "# ACTIVITY=Resolved", "# ACTIVITY=Assigned", "# ACTIVITY=Closed", "# ACTIVITY=Wait - User",
                           "# ACTIVITY=Wait - Implementation", "# ACTIVITY=Wait",
                           "# ACTIVITY=Wait - Vendor", "# ACTIVITY=In Call", "# ACTIVITY=Wait - Customer",
                           "# ACTIVITY=Unmatched", "# ACTIVITY=Cancelled"]
    categorical_features = ["Status", "ACTIVITY", "Involved_ST_Function_Div", "Involved_Org_line_3", "Involved_ST",
                            "SR_Latest_Impact", "Product", "Country", "Owner_Country",
                            "weekday"]

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
                          ('classifier', RandomForestRegressor(n_jobs=7))])
    model = clf.fit(X_train, y_train)

    print("=================== Create DiCE model ===================")
    d_iris = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis="columns"),
                          continuous_features=continuous_features,
                          outcome_name=outcome_name)

    # We provide the type of model as a parameter (model_type)
    m_iris = dice_ml.Model(model=model, backend="sklearn", model_type='regressor')
    method = DICE_METHOD  # genetic, kdtree, random
    # Method random does not support features_weights argument in generate_counterfactuals()
    exp_genetic_iris = Dice(d_iris, m_iris, method=method)
    # exp_genetic_iris = Dice(d_iris, m_iris, method="kdtree")


    # === Load the Transition Graph
    _, transition_graph = transition_system(df, case_id_name=case_id_name, activity_column_name=activity_column_name,
                                            window_size=WINDOW_SIZE)

    print("=================== Create CFEs for all the test cases ===================")

    cfe_before_validation = []
    cfe_after_validation = []
    cfe_not_found = []
    cases_includes_new_data = []
    cases_too_small = []
    i = 0
    for df_test_trace in test_cases:

        query_case_id = get_case_id(df_test_trace)

        if 0 < len(df_test_trace) <= 2:
            print("too small", i, df_test_trace[case_id_name].unique().item())
            cases_too_small.append( get_case_id(df_test_trace, multi=True) )
            continue

        X_test, y_test = prepare_df_for_ml(df_test_trace, outcome_name)
        # Access the last row of the truncated trace to replicate the behavior of a running trace
        total_time_upper_bound = int( y_test.iloc[-1] * (REDUCED_KPI_TIME / 100) )  # A percentage of the original total time of the trace
        query_instances = X_test.iloc[-1:]

        try:
            cfe = generate_cfe( query_instances, total_time_upper_bound, total_cfs=TOTAL_CFS )
            cfe_before_validation.append( (query_case_id, cfe) )

            prefix_of_activities = get_prefix_of_activities(df_single_trace=df_test_trace, window_size=WINDOW_SIZE,
                                                            activity_column_name=activity_column_name)
            cfe_df = validate_transition(cfe, prefix_of_activities=prefix_of_activities, transition_graph=transition_graph,
                                         valid_resources=valid_resources)

            if len(cfe_df) > 0:
                cfe_after_validation.append( (query_case_id, cfe_df) )

        except UserConfigValidationException:
            cfe_not_found.append( query_case_id )
        except TimeoutError as err:  # When function takes too long
            cfe_not_found.append( query_case_id )
            print(f"TimeoutError Occurred: {err}")
        except ValueError:
            # print(f"Includes feature not found in training data: {get_case_id(df_test_trace)}")
            cases_includes_new_data.append( query_case_id )
        except AttributeError as err:
            print(f"The AttributeError: {err}")
        except Exception as err:
            print(f"Broadest Exception handler invoked {err}")

        print(f"Done for idx: {i}, Case: {query_case_id}")

        # For printing results progressively
        if (i % 400) == 0:
            data = {"cfe_before_validation": [len(cfe_before_validation)],
                    "cfe_after_validation": [len(cfe_after_validation)],
                    "cfe_not_found": [len(cfe_not_found)],
                    "cases_includes_new_data": [len(cases_includes_new_data)],
                    "cases_too_small": [len(cases_too_small)]}
            df_result = pd.DataFrame(data)
            df_result.to_csv(RESULTS_FILE_PATH_N_NAME, index=False)

        i += 1
        # if i == 20:
        #     break
        # ----------------------------------------------------------------

    data = {"cfe_before_validation": [len(cfe_before_validation)],
            "cfe_after_validation": [len(cfe_after_validation)],
            "cfe_not_found": [len(cfe_not_found)],
            "cases_includes_new_data": [len(cases_includes_new_data)],
            "cases_too_small": [len(cases_too_small)]}
    df_result = pd.DataFrame(data)
    df_result.to_csv(RESULTS_FILE_PATH_N_NAME, index=False)

    print(f"Time it took: { round( ((time() - start_time) / SECONDS_TO_HOURS), 3) }")
    print("======================================== Testing Complete !!! =============================================")
