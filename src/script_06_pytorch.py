import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

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

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Suppress all warnings
warnings.filterwarnings("ignore")
################################
# Helper Functions
################################
SECONDS_TO_HOURS = 60 * 60
SECONDS_TO_DAYS = 60 * 60 * 24


class ClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNN, self).__init__()  # Means: run the __init__ function of the superclass as well, which in this case is nn.Module
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


def tain_torch_model(model, X, y, epochs=10, lr=0.01):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Loss Function
        criterion = nn.BCELoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Set the model to training mode
        model.train()

        # Convert to PyTorch tensors and create DataLoader
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

            # Print average loss per epoch
            print(f'Epoch [{epoch + 1}/{epochs}], Running Loss: {running_loss / len(dataloader)}, Loss: {loss.item()}')
        print('Finished Training')

        return model


def create_permitted_range():
    # These columns don't exist
    # '# ACTIVITY=Pending Request for acquittance of heirs',
    # '# ACTIVITY=Request completed with customer recovery',
    # '# ACTIVITY=Request completed with account closure',
    # '# ACTIVITY=Request deleted',]

    return {"ACTIVITY": ['Service closure Request with network responsibility',
                  'Service closure Request with BO responsibility',
                  'Pending Request for Reservation Closure',
                  'Pending Liquidation Request',
                  'Request completed with account closure',
                  'Request created',
                  'Authorization Requested',
                  'Evaluating Request (NO registered letter)',
                  'Network Adjustment Requested',
                  'Pending Request for acquittance of heirs',
                  'Request deleted',
                  'Evaluating Request (WITH registered letter)',
                  'Request completed with customer recovery',
                  'Pending Request for Network Information'],
     'time_from_first': [X_train["time_from_first"].min(), X_train["time_from_first"].max()],
     'time_from_previous_et': [X_train["time_from_previous_et"].min(), X_train["time_from_previous_et"].max()],
     'time_from_midnight': [X_train["time_from_midnight"].min(), X_train["time_from_midnight"].max()],
     'activity_duration': [X_train["activity_duration"].min(), X_train["activity_duration"].max()],
     '# ACTIVITY=Service closure Request with network responsibility': [
                                     X_train["# ACTIVITY=Service closure Request with network responsibility"].min(),
                                     X_train["# ACTIVITY=Service closure Request with network responsibility"].max()],
     '# ACTIVITY=Service closure Request with BO responsibility': [
                                         X_train['# ACTIVITY=Service closure Request with BO responsibility'].min(),
                                         X_train['# ACTIVITY=Service closure Request with BO responsibility'].max()],
     '# ACTIVITY=Pending Request for Reservation Closure': [
                                                 X_train['# ACTIVITY=Pending Request for Reservation Closure'].min(),
                                                 X_train['# ACTIVITY=Pending Request for Reservation Closure'].max()],
     '# ACTIVITY=Pending Liquidation Request': [X_train['# ACTIVITY=Pending Liquidation Request'].min(),
                                                X_train['# ACTIVITY=Pending Liquidation Request'].max()],
     '# ACTIVITY=Request created': [X_train['# ACTIVITY=Request created'].min(),
                                    X_train['# ACTIVITY=Request created'].max()],
     '# ACTIVITY=Authorization Requested': [X_train['# ACTIVITY=Authorization Requested'].min(),
                                            X_train['# ACTIVITY=Authorization Requested'].max()],
     '# ACTIVITY=Evaluating Request (NO registered letter)': [
                                                 X_train['# ACTIVITY=Evaluating Request (NO registered letter)'].min(),
                                                 X_train['# ACTIVITY=Evaluating Request (NO registered letter)'].max()],
     '# ACTIVITY=Network Adjustment Requested': [X_train['# ACTIVITY=Network Adjustment Requested'].min(),
                                                 X_train['# ACTIVITY=Network Adjustment Requested'].max()],
     '# ACTIVITY=Back-Office Adjustment Requested': [X_train['# ACTIVITY=Back-Office Adjustment Requested'].min(),
                                                     X_train['# ACTIVITY=Back-Office Adjustment Requested'].max()],
     '# ACTIVITY=Evaluating Request (WITH registered letter)': [
         X_train['# ACTIVITY=Evaluating Request (WITH registered letter)'].min(),
         X_train['# ACTIVITY=Evaluating Request (WITH registered letter)'].max()],
     '# ACTIVITY=Pending Request for Network Information': [
         X_train['# ACTIVITY=Pending Request for Network Information'].min(),
         X_train['# ACTIVITY=Pending Request for Network Information'].max()], }
    # avoiding 'Back-Office Adjustment Requested'


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
    # RESULTS_FILE_PATH_N_NAME = "experiment_results/random-a01-activity_occurrence.csv"  # Default template name
    RESULTS_FILE_PATH_N_NAME = f"experiment_results/{current_file_name.split('.')[0]}.csv"
    configs = {"kpi": "activity_occurrence",
               "window_size": 4,
               # "reduced_kpi_time": 90,                                      # Not used in script_03
               "total_cfs": 50,                                  # Number of CFs DiCE algorithm should produce
               "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),  # genetic, kdtree, random
               "save_load_result_path": RESULTS_FILE_PATH_N_NAME,
               "train_dataset_size": 164_927,                                   # 164_927
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

    case_id_name = 'REQUEST_ID'  # The case identifier column name.
    activity_column_name = "ACTIVITY"
    resource_column_name = "CE_UO"

    data_dir = "./preprocessed_datasets/"
    model_dir = "./ml_models/"
    train_dataset_file = "bank_acc_train.csv"
    # test_dataset_file = "bank_acc_test.csv"
    # test_pickle_dataset_file = "bank_acc_test.pkl"
    test_pickle_dataset_file = "bank_acc_test-500.pkl"
    df = pd.read_csv("./data/bank_account_closure.csv")  # Use full dataset for transition systens
    df_train = pd.read_csv(os.path.join(data_dir, train_dataset_file))
    # df_test = pd.read_csv(os.path.join(data_dir, test_dataset_file))

    # Some Basic Preprocessing
    df = df.fillna("missing")

    # # Temporary
    df_train = df_train[:configs["train_dataset_size"]]
    # df_train = df_train[:5_000]
    ## ---------

    # Unpickle the Standard test-set. To standardize the test across different parameters.
    test_cases = get_test_cases(None, None, load_dataset=True, path_and_filename=os.path.join(data_dir, test_pickle_dataset_file))

    cols_to_vary = [activity_column_name, resource_column_name]
    columns_to_remove = ["START_DATE", "END_DATE", "time_remaining",
                         '# ACTIVITY=Pending Request for acquittance of heirs',
                         '# ACTIVITY=Request completed with customer recovery',
                         '# ACTIVITY=Request completed with account closure',
                         '# ACTIVITY=Request deleted', ]
    outcome_name = "Back-Office Adjustment Requested"

    X_train, y_train = prepare_df_for_ml(df_train, case_id_name, outcome_name, columns_to_remove=columns_to_remove)

    continuous_features = ["time_from_first", "time_from_previous_et", "time_from_midnight", "activity_duration",
                           '# ACTIVITY=Service closure Request with network responsibility',
                           '# ACTIVITY=Service closure Request with BO responsibility',
                           '# ACTIVITY=Pending Request for Reservation Closure',
                           '# ACTIVITY=Pending Liquidation Request',
                           '# ACTIVITY=Request created',
                           '# ACTIVITY=Authorization Requested',
                           '# ACTIVITY=Evaluating Request (NO registered letter)',
                           '# ACTIVITY=Network Adjustment Requested',
                           '# ACTIVITY=Back-Office Adjustment Requested',
                           '# ACTIVITY=Evaluating Request (WITH registered letter)',
                           '# ACTIVITY=Pending Request for Network Information', ]
    categorical_features = ["CLOSURE_TYPE", "CLOSURE_REASON", "ACTIVITY", "CE_UO", "ROLE", "weekday"]

    data_model = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis="columns"),
                              continuous_features=continuous_features,
                              outcome_name=outcome_name)

    # Create One-Hot Encoded data to train and test the ML model
    X_train_ohe = data_model.get_ohe_min_max_normalized_data(X_train)


    # Define the model with the correct input_size
    model = ClassificationNN(X_train_ohe.shape[1])

    # If model exists load else save
    model_path_name = os.path.join(model_dir, current_file_name.split('.')[0]) + ".pth"
    if os.path.exists( model_path_name ):
        model.load_state_dict(torch.load( model_path_name ))
        model.eval()
    else:
        model = tain_torch_model(model, X_train_ohe, y_train, epochs=10)
        # Save model
        torch.save(model.state_dict(), model_path_name)

    print("=================== Create DiCE model ===================")

    # We provide the type of model as a parameter (model_type)
    ml_backend = dice_ml.Model(model=model, backend='PYT',  func="ohe-min-max", model_type='classifier')
    method = configs["dice_method"]
    print("Method", method)
    explainer = Dice(data_model, ml_backend, method=method)

    # === Load activity and resource compatibility thingy
    resource_columns_to_validate = [activity_column_name, resource_column_name]
    valid_resources = activity_n_resources(df, resource_columns_to_validate, threshold_percentage=100)

    # === Load the Transition Graph
    _, transition_graph = transition_system(df, case_id_name=case_id_name, activity_column_name=activity_column_name,
                                            window_size=configs["window_size"], threshold_percentage=100)

    permitted_range = create_permitted_range()
    print("=================== Create CFEs for all the test cases ===================")
    start_from_case = state_obj.run_state["cases_done"]
    for df_test_trace in test_cases[start_from_case:]:
        test_trace_start = time()
        query_case_id = get_case_id(df_test_trace, case_id_name=case_id_name)
        print("--- Start Loop ---,", query_case_id)
        # if 0 < len(df_test_trace) <= 2:
        #     print("too small", cases_done, df_test_trace[case_id_name].unique().item())
        #     result_value = query_case_id
        #     state_obj.add_cfe_to_results(("cases_too_small", result_value))
        #     cases_stored = state_obj.save_state()
        #     cases_done += 1
        #     continue

        X_test, y_test = prepare_df_for_ml(df_test_trace, case_id_name, outcome_name, columns_to_remove=columns_to_remove)

        # # Check if y_test is 0 then don't generate CFE
        # if y_test.iloc[-1] == 0:
        #     result_value = query_case_id
        #     state_obj.add_cfe_to_results(("cases_zero_in_y", result_value))
        #     cases_stored = state_obj.save_state()
        #     cases_done += 1
        #     continue

        # Access the last row of the truncated trace to replicate the behavior of a running trace
        query_instances = X_test.iloc[-1:]
        try:
            cfe = generate_cfe(explainer, query_instances, total_time_upper_bound=None, features_to_vary=cols_to_vary,
                               total_cfs=configs["total_cfs"], kpi=configs["kpi"], proximity_weight=configs["proximity_weight"],
                               sparsity_weight=configs["sparsity_weight"], diversity_weight=configs["diversity_weight"],
                               permitted_range=permitted_range)
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
        if start_from_case >= 2:
            break
        # ----------------------------------------------------------------

    df_result = state_obj.get_run_state_df()
    df_result.to_csv(configs["save_load_result_path"], index=False)

    print(f"Time it took: { round( ((time() - start_time) / SECONDS_TO_HOURS), 3) }")
    print("======================================== Testing Complete !!! =============================================")
