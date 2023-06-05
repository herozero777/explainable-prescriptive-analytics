import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

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
    # RESULTS_FILE_PATH_N_NAME = "experiment_results/random-a01-activity_occurrence.csv"  # Default template name
    RESULTS_FILE_PATH_N_NAME = f"experiment_results/{current_file_name.split('.')[0]}.csv"
    configs = {"kpi": "activity_occurrence",
               "window_size": 3,
               # "reduced_kpi_time": 90,                                      # Not used in script_03
               "total_cfs": 500,                                  # Number of CFs DiCE algorithm should produce
               "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),  # genetic, kdtree, random
               "save_load_result_path": RESULTS_FILE_PATH_N_NAME,
               "train_dataset_size": 170_335,                                   # 170_335
               "proximity_weight": 0.2,
               "sparsity_weight": 0.2,
               "diversity_weight": 5.0,
               "program_run": 0}

    state_obj = StoreTestRun(save_load_path=RESULTS_FILE_PATH_N_NAME)
    save_load_path = state_obj.get_save_load_path()

    if os.path.exists(save_load_path) and args.first_run:
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
    df_train = df_train[:configs["train_dataset_size"]]
    ## ---------

    resource_columns_to_validate = [activity_column_name, 'CE_UO', 'ROLE']
    valid_resources = activity_n_resources(df, resource_columns_to_validate)
    # len(valid_resources)

    # Unpickle the Standard test-set. To standardize the test across different parameters.
    test_cases = get_test_cases(None, None, load_dataset=True, path_and_filename=os.path.join(data_dir, test_pickle_dataset_file))

    cols_to_vary = ["ACTIVITY", "CE_UO", "ROLE"]

    outcome_name = "Back-Office Adjustment Requested"

    X_train, y_train = prepare_df_for_ml(df_train, case_id_name, outcome_name, columns_to_remove=["START_DATE", "END_DATE", "time_remaining"])

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
    data_model = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis="columns"),
                          continuous_features=continuous_features,
                          outcome_name=outcome_name)

    # We provide the type of model as a parameter (model_type)
    ml_backend = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
    method = configs["dice_method"]
    explainer = Dice(data_model, ml_backend, method=method)

    # === Load the Transition Graph
    _, transition_graph = transition_system(df, case_id_name=case_id_name, activity_column_name=activity_column_name,
                                            window_size=configs["window_size"])

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

        X_test, y_test = prepare_df_for_ml(df_test_trace, case_id_name, outcome_name, columns_to_remove=["START_DATE", "END_DATE", "time_remaining"])

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
            cfe = generate_cfe(explainer, query_instances, total_time_upper_bound=None, features_to_vary=cols_to_vary,
                               total_cfs=configs["total_cfs"], KPI=configs["kpi"], proximity_weight=configs["proximity_weight"],
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
            df_result.to_csv(configs["save_load_result_path"] , index=False)

        cases_done += 1
        # if i >= 20:
        #     break
        # ----------------------------------------------------------------

    df_result = state_obj.get_run_state_df()
    df_result.to_csv(configs["save_load_result_path"], index=False)

    print(f"Time it took: { round( ((time() - start_time) / SECONDS_TO_HOURS), 3) }")
    print("======================================== Testing Complete !!! =============================================")
