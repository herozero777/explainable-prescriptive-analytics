import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.transition_system import transition_system, indexs_for_window, list_to_str
from src.function_store import StoreTestRun, extract_algo_name, generate_cfe, get_case_id, prepare_df_for_ml, \
    activity_n_resources, get_test_cases, get_prefix_of_activities, validate_transition, get_cut_test_traces

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from time import time
import argparse
import os
import numpy as np
import pandas as pd
import random
import warnings

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Suppress all warnings
warnings.filterwarnings("ignore")
################################
# Helper Functions
################################
SECONDS_TO_HOURS = 60 * 60
SECONDS_TO_DAYS = 60 * 60 * 24


def save_experiment_info(experiment_results_dir, nn_results_filename = "nn_results.csv", config_dict = None):
    """
    Save the configurations used for the experiments in nn_results_filename file. If is exists, append the new results
    as the last row else create a new file with the results.
    Args:
        experiment_results_dir:
        nn_results_filename:
        config_dict:

    Returns:
        None
    """
    if config_dict is None:
        raise ValueError("config_dict is None, Provide a dictionary with the configurations used for the experiment")
    if not os.path.exists(experiment_results_dir):
        os.makedirs(experiment_results_dir)
    if not os.path.exists(os.path.join(experiment_results_dir, nn_results_filename)):
        df = pd.DataFrame(columns=config_dict.keys())
        df.to_csv(os.path.join(experiment_results_dir, nn_results_filename), index=False)
    df = pd.read_csv(os.path.join(experiment_results_dir, nn_results_filename))
    df = df.append(config_dict, ignore_index=True)
    df.to_csv(os.path.join(experiment_results_dir, nn_results_filename), index=False)


def random_lr(log_lower = -3, log_upper = -1):
    """
    Generate a random learning rate within the range [log_lower, log_upper].
    Set the range boundaries
    Args:
        log_lower: if -3 = 10^-3 = 0.001
        log_upper: if -1 = 10^-1 = 0.1

    Returns:
        learning_rate: a random learning rate within the range
    """
    # Generate a random learning rate within the range
    log_lr = random.uniform(log_lower, log_upper)
    learning_rate = 10 ** log_lr
    return learning_rate


class ClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNN, self).__init__()  # Means: run the __init__ function of the superclass as well, which in this case is nn.Module
        self.layer1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        # self.layer3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.layer6 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.layer7 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, inference=True):
        """ Cuz Dice Needs to use the model later and by default it expects the output to be from a sigmoid function"""
        x = self.layer1(x)
        x = self.relu( self.bn1(x) )
        x = self.relu(self.bn2(self.layer2(x)))
        # x = self.relu(self.bn3(self.layer3(x)))
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.relu(self.bn5(self.layer5(x)))
        x = self.relu(self.bn6(self.layer6(x)))
        x = self.layer7(x)  # no sigmoid here for training
        if inference:
            x = self.sigmoid(x)
        return x


def tain_torch_model(model, X, y, X_test, y_test, epochs=10, lr=0.01, weight_decay=0.01, batch_size=64, experiment_name="exp_01",
                     random_state=42):
    if isinstance(X, pd.DataFrame):
        X = X.values
        y = y.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
        y_test = y_test.values.astype(int)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with L2 regularization (weight decay)
    # The value for weight_decay will need to be chosen carefully,
    # as too large a value might cause all weights to vanish,
    # and too small might not have a noticeable effect
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define SummaryWriter
    writer = SummaryWriter(f'nn_runs/{experiment_name}')

    # Assume X is your feature data and y are the labels
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=random_state)

    # Compute label weights in training dataset samples weights
    class_sample_count = np.unique(y_train, return_counts=True)[1]
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[y_train]

    # Create sampler and dataloader
    # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    # dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # For Measuring F1-score
    X_train_gpu = torch.tensor(X_train, dtype=torch.float).to(device)
    X_valid_gpu = torch.tensor(X_valid, dtype=torch.float).to(device)
    X_test_gpu = torch.tensor(X_test, dtype=torch.float).to(device)

    # Convert to PyTorch tensors and create DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float), torch.tensor(y_valid, dtype=torch.float))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)

    # Training loop
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        avg_loss = 0.0
        for inputs, labels in train_loader:
            # Send data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, inference=False).flatten()
            # loss = criterion(outputs, labels.unsqueeze(-1))
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Print statistics
            avg_loss += loss.item()

        avg_train_loss = avg_loss / len(train_loader)
        writer.add_scalar('training loss', avg_train_loss, epoch)

        # Print average loss per epoch
        # print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss}, Loss: {loss.item()}')

        # Validation loss
        model.eval()
        avg_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, inference=False).flatten()
                loss = criterion(outputs, labels)
                avg_loss += loss.item()

        avg_valid_loss = avg_loss / len(valid_loader)
        writer.add_scalar('validation loss', avg_valid_loss, epoch)

        # if epoch % 10 == 0:   # Use later
        # Log Train F1-score
        writer.add_scalar('Train F1-score', f1_score(y_train, model(X_train_gpu).detach().cpu().numpy().round()), epoch)
        # Log Validation F1-score
        writer.add_scalar('Valid F1-score', f1_score(y_valid, model(X_valid_gpu).detach().cpu().numpy().round()), epoch)
        # Log Test F1-score
        writer.add_scalar('Test F1-score', f1_score(y_test, model(X_test_gpu).detach().cpu().numpy().round()), epoch)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Variables to configure ======
    # Uses DiCE algo method as the first word of the .csv file.
    # Just for running these many experiments. Going to duplicate this as template. Each experiment run will have its
    # own script copy. This creates many duplicate files, but its allows to spot the experiment name in the `htop` tool.
    # RESULTS_FILE_PATH_N_NAME = "experiment_results/random-a01-activity_occurrence.csv"  # Default template name
    RESULTS_FILE_PATH_N_NAME = f"experiment_results/{current_file_name.split('.')[0]}.csv"
    configs = {"batch_size": 256,
               "learning_rate": random_lr(),
               "epochs": 4_000,                                                   # Training epochs for the NN model
               "experiment_name": current_file_name.split('.')[0],                # Name of results dir in tensorboard
               "split_random_state": random.randint(1, 1000),
               "weight_decay": 0.009,
               "L2_regularization": True,
               "Batch_Normalization": True,
               "program_run": 0,
               "Layers": [512, 256, 64, 32, 16, 1]
               }

    state_obj = StoreTestRun(save_load_path=RESULTS_FILE_PATH_N_NAME)
    save_load_path = state_obj.get_save_load_path()

    if os.path.exists(save_load_path) and args.first_run:  # TODO: This check is not thorought enough, improve it.
        raise FileExistsError(f"This is program's first run yet the pickle file: {save_load_path} exists. Please remove"
                              f"it to run it with the flag --first_run")

    # ==== If saved progress exists, load it.
    # if os.path.exists(save_load_path):
    #     state_obj.load_state()
    #     cases_done = state_obj.run_state["cases_done"]
    #     configs = state_obj.get_model_configs()
    #     configs['program_run'] += 1
    #     print(f"Run: {configs['program_run']} of {configs['save_load_result_path'].split('/')[1] }")
    # else:
    #     configs['program_run'] += 1
    #     print(f"Run: {configs['program_run']} of {configs['save_load_result_path'].split('/')[1] }")
    #     state_obj.add_model_configs(configs=configs)
    #     cases_done = 0

    case_id_name = 'REQUEST_ID'  # The case identifier column name.
    activity_column_name = "ACTIVITY"
    resource_column_name = "CE_UO"

    data_dir = "./preprocessed_datasets/"
    model_dir = "./ml_models/"
    experiment_results_dir = "./experiment_results/"
    train_dataset_file = "bank_acc_train.csv"
    # test_dataset_file = "bank_acc_test.csv"
    test_pickle_dataset_file = "bank_acc_test.pkl"
    df = pd.read_csv("./data/bank_account_closure.csv")  # Use full dataset for transition systens
    df_train = pd.read_csv(os.path.join(data_dir, train_dataset_file))
    # df_test = pd.read_csv(os.path.join(data_dir, test_dataset_file))
    # Unpickle the Standard test-set. To standardize the test across different parameters.
    test_cases = get_test_cases(None, None, load_dataset=True, path_and_filename=os.path.join(data_dir, test_pickle_dataset_file))
    df_test = get_cut_test_traces(test_cases)

    # Save Model Config
    print("Configs:", configs)
    # ToDo (If later results can also be stored in the same row)
    save_experiment_info(experiment_results_dir, nn_results_filename="nn_results.csv", config_dict=configs)

    # Some Basic Preprocessing
    df = df.fillna("missing")

    cols_to_vary = [activity_column_name, resource_column_name]
    columns_to_remove = ["START_DATE", "END_DATE", "time_remaining",
                         '# ACTIVITY=Pending Request for acquittance of heirs',
                         '# ACTIVITY=Request completed with customer recovery',
                         '# ACTIVITY=Request completed with account closure',
                         '# ACTIVITY=Request deleted', ]
    outcome_name = "Back-Office Adjustment Requested"

    X_train, y_train = prepare_df_for_ml(df_train, case_id_name, outcome_name, columns_to_remove=columns_to_remove)
    X_test, y_test = prepare_df_for_ml(df_test, case_id_name, outcome_name, columns_to_remove=columns_to_remove)

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
    X_test_ohe = data_model.get_ohe_min_max_normalized_data(X_test)

    # Define the model with the correct input_size
    model = ClassificationNN(X_train_ohe.shape[1])

    model_path_name = os.path.join(model_dir, current_file_name.split('.')[0]) + ".pth"

    model = model.to(device)
    model = tain_torch_model(model, X_train_ohe, y_train, X_test_ohe, y_test, epochs=configs["epochs"],
                             lr=configs["learning_rate"], weight_decay=configs["weight_decay"],
                             batch_size=configs["batch_size"],
                             experiment_name=configs["experiment_name"], random_state=configs["split_random_state"])
    model = model.to('cpu')
    # Save model
    torch.save(model.state_dict(), model_path_name)

    #
    # if os.path.exists( model_path_name ):
    #     print(f"Model exists loading model")
    #     model.load_state_dict(torch.load( model_path_name ))
    #     model.eval()
    # else:
    #     print(f"Model does not exist, training model")
    #     model = model.to(device)
    #     model = tain_torch_model(model, X_train_ohe, y_train, epochs=configs["epochs"])
    #     model = model.to('cpu')
    #     # Save model
    #     torch.save(model.state_dict(), model_path_name)

    print("======================================== Testing Complete !!! =============================================")
