# Prescriptive-Analytics

### Install requirements 
Works on Python (3.9 & 3.10). Make sure you have folder `experiment_results` for running the notebooks and the scripts 
in `src` directory.
```bash
pip install -r req.txt
pip install -r requirements.txt
```

### Parameters
Mandatory parameters: \
--filename_completed --> log file containing all completed cases \
--case_id_name, activity_name, start_date_name --> self explanatory, name of the columns containing the specified information \
--pred_column --> kpi to be predicted (remaining_time | lead_time | total_cost | independent_activity) 

Optional parameters: \
--end_date_name \
--resource_name, role_name \
--predict_activities --> name of the activity to be predcited (mandatory if pred_column is independent_activity) \
--experiment_name --> folder in which results will be saved at the end of the experiment \
--override (default True) --> deletes temporary files from previous run \
--shap (default False) --> if you want to calculate also the explanations of predictions \
--explain (default False) --> if you want to calculate the explanations of recommendations
--outlier_thrs (default 0.001) --> if you want to set the threshold for outlier's frequence

## Code Example (run it from shell, if you have a Windows machine, remove the "\" and run the complete command )

## Recommendations and explanations for VINST case study on total time
python main_recsys.py --filename_completed 'data/VINST cases incidents.csv' --case_id_name SR_Number --activity_name ACTIVITY --start_date_name Change_Date+Time --pred_column lead_time --resource_name Involved_ST --role_name Involved_Org_line_3 --experiment_name exp_time_VINST --explain True

### Recommendations and explanations for BAC case study on Activity "Pending Liquidation Request"
python main_recsys.py --filename_completed data/completed.csv --case_id_name REQUEST_ID --activity_name ACTIVITY --start_date_name START_DATE --resource_name CE_UO --role_name ROLE --end_date_name END_DATE --pred_column independent_activity --predict_activities "Pending Liquidation Request" --experiment_name prova_activity_BAC_PLR --explain True

## Notebooks
Run the `jupyter` server in the directory: `explainable-prescriptive-analytics` the project directory, it shouldn't be 
running in the `src` directory.

### 00_preprocess_event_logs.ipynb
This notebook preprocess the event-log data, so that machine learning can be applied to it. The notebook currently 
preprocesses the data for 2 KPIs "Activity Occurrence" and "Total Time".

### 02_DiCE_kpi-time.ipynb
This notebook contains code that uses VINST dataset and tries to optimize the `lead_time` aka the total
time it takes to complete a trace. DiCE algorithm is used for generating counterfactual explanations (CFEs)
and then validation of the CFEs is done as postprocessing steps.

## Test Code
#### src/transition_system.py
To test the transition system use the following command. Make sure to install pytest library. 

    pytest src/transition_system.py


---
## Info About Datasets
### VINST Dataset
Columns
- START_DATE - Start time of the activity. Format is Timestamp in milliseconds.
- END_DATE - End time of the activity. Format is Timestamp in milliseconds.


## Contributors
Riccardo Galanti
Alessandro Padella
Mohammad Ismail Tirmizi
