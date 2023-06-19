import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from src.transition_system import transition_system, indexs_for_window, list_to_str
from src.function_store import StoreTestRun, extract_algo_name, generate_cfe, get_case_id, prepare_df_for_ml, \
    activity_n_resources, get_test_cases, get_prefix_of_activities, validate_transition

from datetime import datetime
from time import time
import argparse
import os
import pandas as pd
import random
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
################################
# Helper Functions
################################
SECONDS_TO_HOURS = 60 * 60
SECONDS_TO_DAYS = 60 * 60 * 24


if __name__ == '__main__':
    start_time = time()
    ################################
    # Parse command line arguments
    ################################
    parser = argparse.ArgumentParser(description='Script for Testing DiCE algorithm. The script runs the algorithm with'
                                                 'desired configuration on a test dataset.')
    parser.add_argument('--first_run', action='store_true', help="Use this flag if this is the first time running the script.")
    # TODO: add a flag to denote "use file name as the method name mode"
    args = parser.parse_args()

    print(f"========================= Program Start at: {datetime.fromtimestamp(start_time)} =========================")
    # Get the path of the current script file
    script_path = os.path.dirname(os.path.abspath(__file__))
    print("Current Working Directory:", os.getcwd())
    print(script_path)
    current_file_name = os.path.basename(__file__)
    print("File name:", current_file_name)

    # ====== Variables to configure ======
    # Uses DiCE algo method as the first word of the .csv file.
    # Just for running these many experiments. Going to duplicate this as template. Each experiment run will have its
    # own script copy. This creates many duplicate files, but its allows to spot the experiment name in the `htop` tool.
    # RESULTS_FILE_PATH_N_NAME = "experiment_results/random-t01-total_time.csv"  # Default template name
    RESULTS_FILE_PATH_N_NAME = f"experiment_results/{current_file_name.split('.')[0]}.csv"
    configs = {"kpi": "total_time",                             # "activity_occurrence", "total_time"
               "window_size": 3,
               "reduced_kpi_time": 90,
               "total_cfs": 50,                                  # Number of CFs DiCE algorithm should produce
               "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),  # genetic, kdtree, random
               "save_load_result_path": RESULTS_FILE_PATH_N_NAME,
               "train_dataset_size": 39_375,                                   # 39_375
               "proximity_weight": 0.2,
               "sparsity_weight": 0.2,
               "diversity_weight": 5.0,
               "program_run": 0}

    state_obj = StoreTestRun(save_load_path=RESULTS_FILE_PATH_N_NAME)
    save_load_path = state_obj.get_save_load_path()

    if os.path.exists(save_load_path) and args.first_run:  # TODO: This check is not thorought enough, improve it.
        raise FileExistsError(f"This is program's first run yet the pickle file: {save_load_path} exists. Please remove"
                              f"it to run it with the flag --first_run")

    # ==== If saved progress exists, load it.
    if os.path.exists(save_load_path):
        state_obj.load_state()
        cases_done = state_obj.run_state["cases_done"]
        configs = state_obj.get_model_configs()
        configs['program_run'] += 1
        print(f"Run: {configs['program_run']} of {configs['save_load_result_path'].split('/')[1] }")
    else:
        configs['program_run'] += 1
        print(f"Run: {configs['program_run']} of {configs['save_load_result_path'].split('/')[1] }")
        state_obj.add_model_configs(configs=configs)
        cases_done = 0

    print("Configs:", configs)

    case_id_name = 'SR_Number'  # The case identifier column name.
    start_date_name = 'Change_Date+Time'  # Maybe change to start_et (start even time)
    activity_column_name = "ACTIVITY"

    data_dir = "./preprocessed_datasets/"
    train_dataset_file = "vinst_train.csv"
    # test_dataset_file = "vinst_test.csv"
    test_pickle_dataset_file = "vinst_test.pkl"
    df = pd.read_csv("./data/VINST cases incidents.csv")  # Use full dataset for transition systens
    df_train = pd.read_csv(os.path.join(data_dir, train_dataset_file))
    # df_test = pd.read_csv(os.path.join(data_dir, test_dataset_file))

    # Some Basic Preprocessing
    df = df.fillna("missing")

    # # Temporary
    df_train = df_train[:configs["train_dataset_size"]]
    ## ---------

    # Unpickle the Standard test-set. To standardize the test across different parameters.
    test_cases = get_test_cases(None, None, load_dataset=True, path_and_filename=os.path.join(data_dir, test_pickle_dataset_file))

    cols_to_vary = ["ACTIVITY", "Involved_ST_Function_Div", "Involved_Org_line_3", "Involved_ST"]

    outcome_name = "lead_time"

    X_train, y_train = prepare_df_for_ml(df_train, case_id_name, outcome_name, columns_to_remove=["Change_Date+Time", "time_remaining"])

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
    data_model = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis="columns"),
                              continuous_features=continuous_features,
                              outcome_name=outcome_name)

    # We provide the type of model as a parameter (model_type)
    ml_backend = dice_ml.Model(model=model, backend="sklearn", model_type='regressor')
    method = configs["dice_method"]
    explainer = Dice(data_model, ml_backend, method=method)

    # === Load activity and resource compatibility thingy
    resource_columns_to_validate = [activity_column_name, 'Involved_ST_Function_Div', 'Involved_Org_line_3',
                                    'Involved_ST', 'Country', 'Owner_Country']
    valid_resources = activity_n_resources(df, resource_columns_to_validate, threshold_percentage=100)

    # === Load the Transition Graph
    _, transition_graph = transition_system(df, case_id_name=case_id_name, activity_column_name=activity_column_name,
                                            window_size=configs["window_size"], threshold_percentage=100)

    print("=================== Create CFEs for all the test cases ===================")
    start_from_case = state_obj.run_state["cases_done"]
    for df_test_trace in test_cases[start_from_case:]:
        test_trace_start = time()
        query_case_id = get_case_id(df_test_trace)
        print("--- Start Loop ---,", query_case_id)
        # if 0 < len(df_test_trace) <= 2:
        #     print("too small", cases_done, query_case_id)
        #     result_value = query_case_id
        #     state_obj.add_cfe_to_results(("cases_too_small", result_value))
        #     cases_stored = state_obj.save_state()
        #     cases_done += 1
        #     continue

        X_test, y_test = prepare_df_for_ml(df_test_trace, case_id_name, outcome_name, columns_to_remove=["Change_Date+Time", "time_remaining"])
        # Access the last row of the truncated trace to replicate the behavior of a running trace
        query_instances = X_test.iloc[-1:]
        total_time_upper_bound = int( y_test.iloc[-1] * ( configs["reduced_kpi_time"] / 100) )  # A percentage of the original total time of the trace

        try:
            cfe = generate_cfe(explainer, query_instances, total_time_upper_bound, features_to_vary=cols_to_vary,
                               total_cfs=configs["total_cfs"], kpi=configs["kpi"], proximity_weight=configs["proximity_weight"],
                               sparsity_weight=configs["sparsity_weight"], diversity_weight=configs["diversity_weight"])
            result_value = (query_case_id, cfe)
            state_obj.add_cfe_to_results(("cfe_before_validation", result_value))  # save after cfe validation

            prefix_of_activities = get_prefix_of_activities(df_single_trace=df_test_trace, window_size=configs["window_size"],
                                                            activity_column_name=activity_column_name)
            cfe_df = validate_transition(cfe, prefix_of_activities=prefix_of_activities, transition_graph=transition_graph, valid_resources=valid_resources,
                                         activity_column_name=activity_column_name, resource_columns_to_validate=resource_columns_to_validate)

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
        # except Exception as err:
        #     print(f"Broadest Exception handler invoked", err)
        #     state_obj.add_cfe_to_results(("exceptions", query_case_id))
        #     cases_stored = state_obj.save_state()

        # For printing results progressively
        if (cases_done % 100) == 0:
            df_result = state_obj.get_run_state_df()
            df_result.to_csv(configs["save_load_result_path"], index=False)

        print(f"Time it took: {round(((time() - test_trace_start) / 60), 3)}")
        cases_done += 1
        # if i >= 20:
        #     break
        # ----------------------------------------------------------------

    df_result = state_obj.get_run_state_df()
    df_result.to_csv(configs["save_load_result_path"], index=False)

    print(f"Time it took: { round( ((time() - start_time) / SECONDS_TO_HOURS), 3) }")
    print("======================================== Testing Complete !!! =============================================")
