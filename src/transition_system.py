import pandas as pd
import numpy as np
import os
from datetime import datetime
pd.options.display.max_columns= None
from collections import Counter


def indexs_for_window(current_index=0, window_size=2, end_exclusive=True):
    """
    Designed to access a dataframe in a windowed fashion, meaning if you want to access last n elements from a given
    index position. E.g you want to access previous 2 elements from any index. So for index values going like:
    0, 1, 2, 3, 4 the function will return [0,1], [0,2], [0,3], [1,4], [2,5]
    Args:
        current_index: The index position in the dataframe.
        window_size: Size of the window that is created by this function.
        end_exclusive: Normally this is the case, example with list slicing in python. But sometimes like with pandas
                      .loc the end_index can be inclusive as well. In that case set `False`.
    Returns:
        Tuple[int, int]
    """
    # === For test
    # end_correction = 1 if end_exclusive else 0
    # zero_correction = 1
    # for current_index in range(0, 5):
    #     print( max(current_index - window_size + zero_correction, 0), current_index + end_correction)
    end_correction = 1 if end_exclusive else 0
    zero_correction = 1
    start_index = max(current_index - window_size + zero_correction, 0)
    end_index = current_index + end_correction
    return start_index, end_index


class TestIndexForWindow:
    def test_end_exclusive_true(self):
        end_exclusive = True
        # Test cases
        # 0:1
        # 0:2
        # 0:3
        # 1:4
        # 2:5

        start_index, end_index = indexs_for_window(current_index=0, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (0, 1)
        start_index, end_index = indexs_for_window(current_index=1, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (0, 2)
        start_index, end_index = indexs_for_window(current_index=2, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (0, 3)
        start_index, end_index = indexs_for_window(current_index=3, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (1, 4)
        start_index, end_index = indexs_for_window(current_index=4, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (2, 5)

    def test_end_exclusive_false(self):
        end_exclusive = False
        # Test cases
        # 0:0
        # 0:1
        # 0:2
        # 1:3
        # 2:4

        start_index, end_index = indexs_for_window(current_index=0, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (0, 0)
        start_index, end_index = indexs_for_window(current_index=1, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (0, 1)
        start_index, end_index = indexs_for_window(current_index=2, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (0, 2)
        start_index, end_index = indexs_for_window(current_index=3, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (1, 3)
        start_index, end_index = indexs_for_window(current_index=4, window_size=3, end_exclusive=end_exclusive)
        assert (start_index, end_index) == (2, 4)

def list_to_str(list_of_strings ):
    """ Converts ['A', 'B', 'C'] -> 'A, B, C' """
    return ", ".join(list_of_strings)


def transition_system(df, case_id_name=None, activity_column_name="ACTIVITY", threshold_percentage=100,
                      use_symbols=False, window_size=3):
    """
    Creates a transition graph from traces. The traces are broken into prefixes and for each prefix the next possible
    activity is recorded. (ref: Explainable Process Prescriptive Analytics, Fig 1). E.g. prefix  <a> has [b, c, e] as
    the next possible activities but prefix <a, b> has [c, d, f] as the next activities.
    Args:
        df (pd.DataFrame):
        case_id_name:
        activity_column_name:
        threshold_percentage (float): Threshold value
        use_symbols (Bool): If all activities be mapped to symbols and those be used instead. `True` means yes do that.
                            When True it doesn't use the `activity_column_name` variable so its value doesn't matter.
        window_size (int): Max number of prefixes to keep in the transition system. E.g. For `window_size` = 3 if the
                          prefix is <a, b, c, d>, the transition system will consider prefix <b, c, d> and add next
                          possible activities to its list.

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

        activity_col = "activity_symbols"
    else:
        activity_col = activity_column_name

    threshold = threshold_percentage / 100
    gdf = df.groupby(case_id_name)
    activity_paths_count = {}
    # Iterate over each trace separately. Achieved by GROUPBY on case-ids.
    for case_id, group in gdf:
        trace_path = list_to_str( group[activity_col].to_list() )
        if activity_paths_count.get( trace_path ):
            activity_paths_count[ trace_path ] += 1
        else:
            activity_paths_count[ trace_path ] = 1

    paths_and_their_counts = [ (k, v) for k, v in activity_paths_count.items() ]
    sorted_paths_and_counts = sorted( paths_and_their_counts, key=lambda item: item[1], reverse=True )
    sorted_paths = [path for path, count in sorted_paths_and_counts ]
    amount_of_paths_to_select = int( len(sorted_paths) * threshold ) + 1
    high_frequency_paths = sorted_paths[:amount_of_paths_to_select]

    # === Create the transition System
    activity_col_position = df.columns.get_loc(activity_col)
    # For edge case: Where the first activity doesn't have a prefix, so create an empty prefix
    transition_graph[""] = set()
    # Iterate over each trace separately. Achieved by GROUPBY on case-ids.
    for case_id, group in gdf:

        previous_activity_str = ""
        # Don't run for traces with low frequency paths
        if list_to_str(group[activity_col]) not in high_frequency_paths:
            continue

        for idx, (_, row) in enumerate( group.iterrows() ):

            start_index, end_index = indexs_for_window(idx, window_size=window_size, end_exclusive=True)
            # print(start_index, end_index)
            # print(group.iloc[start_index:end_index, activity_col_position])
            activities_str = list_to_str(group.iloc[start_index:end_index, activity_col_position].to_list())
            if activities_str not in transition_graph.keys():
                transition_graph[activities_str] = set()

            # if idx != 0:  # First activity is not added to any prefix
            if row[activity_col] not in transition_graph[previous_activity_str]:
                transition_graph[previous_activity_str].add( row[activity_col] )

            previous_activity_str = activities_str
        #
        # z += 1
        # if z == 5:
        #     break
    return df, transition_graph


if __name__ == '__main__':
    data_dir = "./data"

    dataset = "completed.csv"               # bank_account_closure
    dataset = "VINST cases incidents.csv"   # VINST dataset
    data_file_path = os.path.join(data_dir, dataset)

    activity_column_name = "ACTIVITY"
    if dataset == "completed.csv":
        case_id_name = "REQUEST_ID"
        start_date_name = "START_DATE"
        resource_column_name = "CE_UO"
        df = pd.read_csv(data_file_path)  # concern: what is date col position is different?
        df[start_date_name] = pd.to_datetime(df.iloc[:, 5], unit='ms')

    elif dataset == "VINST cases incidents.csv":
        case_id_name = 'SR_Number'  # The case identifier column name.
        start_date_name = 'Change_Date+Time'  # Maybe change to start_et (start even time)
        df = pd.read_csv(data_file_path)  # concern: what is date col position is different?
        df[start_date_name] = pd.to_datetime(df.iloc[:, 1])

    unique_activities = df[activity_column_name].unique()

    # Create a dictionary that associates a unique symbol to each unique value
    symbol_dict = {}
    for index, value in enumerate(unique_activities):
        symbol_dict[value] = chr(ord('A') + index)

    # Map each value in the list to its corresponding unique symbol
    symbol_list = [symbol_dict[value] for value in df[activity_column_name]]
    df["activity_symbols"] = symbol_list

    ################################
    # Activity Transition System
    ################################
    df, transition_graph = transition_system(df, case_id_name, use_symbols=True)

    ################################
    # Block of code that creates pair of valid combinations of activity & resource, so that later new combinations
    # can be validated.
    ################################
    # activity_resource_pair is a set of activity symbols and resource tuples.
    # E.g. { (act1, res1), ..., (act6, res9) }


    activity_resource_pair = set(zip(df["activity_symbols"], df[resource_column_name]))

    # To test if a pair of activity and resource is valid
    assert ('F', '00870') in activity_resource_pair
    print("Test passed")
    assert not ('F', '1100870') in activity_resource_pair
    print("Test passed")

    resource_column_names = [activity_column_name, 'Involved_ST_Function_Div', 'Involved_Org_line_3', 'Involved_ST', 'SR_Latest_Impact', 'Country', 'Owner_Country']
    valid_resource_combo = set(df[resource_column_names].apply(tuple, axis='columns'))