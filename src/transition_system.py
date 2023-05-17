import pandas as pd
import numpy as np
import os
from datetime import datetime
pd.options.display.max_columns= None


# TODO: Just create a function that returns a hashmap for next possible activity
# TODO: a function that returns a pair of valid activity and resources
def transition_system(df, case_id_name=None, activity_column_name="ACTIVITY", thrs=1.0,
                      use_symbols=False):
    """
    Args:
        df (pd.DataFrame):
        case_id_name:
        activity_column_name:
        thrs (float): Threshold value
        use_symbols (Bool): If all activities be mapped to symbols and those be used instead. `True` means yes do that.

    Raises:
         AssertError: if unique activities are more than 26

    Returns:
        Tupe[pd.DataFrame, dict]: first element is the new dataframe, second is the transition graph
    """

    if case_id_name is None:
        raise TypeError("Case id name is missing! please specify it.")

    transition_graph = {}
    unique_activities = df[activity_column_name].unique()

    if use_symbols:
        # Limit to 26, cuz not sure if ASCII characters after 'Z' are safe to use
        assert len(unique_activities) <= 26, "The number of unique activities is more than 26"

        # Create a dictionary that associates a unique symbol to each unique value
        symbol_dict = {}
        for index, value in enumerate(unique_activities):
            symbol_dict[value] = chr(ord('A') + index)

        # Map each value in the list to its corresponding unique symbol
        symbol_list = [symbol_dict[value] for value in df[activity_column_name]]
        df["activity_symbols"] = symbol_list

        # Create list for each symbol (activity) this list will represent possible next activities
        for symbol in set(symbol_list):
            transition_graph[symbol] = set()

        activity_col = "activity_symbols"
    else:

        # Create list for each activity this list will represent possible next activities
        for activity in unique_activities:
            transition_graph[activity] = set()

        activity_col = activity_column_name

    # === Create the transition System
    z = 0
    gdf = df.groupby(case_id_name)
    # Iterate over each trace separately. Achieved by GROUPBY on case-ids.
    for case_id, group in gdf:
        # print(group[ [activity_column_name, "activity_symbols"] ])

        prev_symbol = None
        group_idx = 0
        for _, row in group.iterrows():
            # print(f"group i: {group_idx}")
            if group_idx != 0:
                if row[activity_col] not in transition_graph.get(prev_symbol):
                    transition_graph.get(prev_symbol).add( row[activity_col] )
            # # =================================================================
            # # Creates a set of next valid (activity & resource). But it looks like validating
            # # activity and resource separately is more efficient.
            # if group_idx != 0:
            #     if (row[activity_col], row[resource_column_name]) not in transition_graph.get(prev_symbol):
            #         transition_graph.get(prev_symbol).add( (row[activity_col], row[resource_column_name]) )
            # # =================================================================
            prev_symbol = row[activity_col]
            group_idx += 1
        #
        # z += 1
        # if z == 5:
        #     break
    return df, transition_graph

if __name__ == '__main__':
    dataset = "completed.csv"  # bank_account_closure

    data_dir = "./data"
    data_file_path = os.path.join(data_dir, dataset)

    case_id_name = "REQUEST_ID"
    start_date_name = "START_DATE"
    activity_column_name = "ACTIVITY"
    resource_column_name = "CE_UO"

    df = pd.read_csv(data_file_path)  # concern: what is date col position is different?

    df[start_date_name] = pd.to_datetime(df.iloc[:, 5], unit='ms')

    unique_activities = df[activity_column_name].unique()

    # Create a dictionary that associates a unique symbol to each unique value
    symbol_dict = {}
    for index, value in enumerate(unique_activities):
        symbol_dict[value] = chr(ord('A') + index)

    # Map each value in the list to its corresponding unique symbol
    symbol_list = [symbol_dict[value] for value in df[activity_column_name]]
    df["activity_symbols"] = symbol_list

    """
    Block of code that creates pair of valid combinations of activity & resource, so that later new combinations 
    can be validated.
    """
    # activity_resource_pair is a set of activity symbols and resource tuples.
    # E.g. { (act1, res1), ..., (act6, res9) }
    activity_resource_pair = set(zip(df["activity_symbols"], df[resource_column_name]))

    # To test if a pair of activity and resource is valid
    assert ('F', '00870') in activity_resource_pair
    print("Test passed")
    assert not ('F', '1100870') in activity_resource_pair
    print("Test passed")
