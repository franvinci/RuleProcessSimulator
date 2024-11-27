from pm4py.discovery import discover_petri_net_inductive
import src.simulator as simulator

import numpy as np
import pandas as pd
import pm4py

import os


def discovery_and_simulate(
        log, net=None, initial_marking=None, final_marking=None, 
        max_depths=range(1,6), 
        noise_threshold_im=0.0, 
        start_ts_simulation=None, 
        label_data_attributes=[], label_data_attributes_categorical=[], 
        n_sim_traces=1000, n_sim=5
        ):

    print('DISCOVERY...')
    if not net:
        print("Process model discovery...")
        net, initial_marking, final_marking = discover_petri_net_inductive(log, noise_threshold=noise_threshold_im)

    parameters = simulator.SimulatorParameters(net, initial_marking, final_marking)
    parameters.discover_from_eventlog(
        log, 
        max_depths_cv=max_depths,
        label_data_attributes=label_data_attributes, 
        label_data_attributes_categorical=label_data_attributes_categorical, 
        mode_history_weights=True
        )

    sim_logs = []
    for i in range(1, n_sim+1):
        simulator_eng = simulator.SimulatorEngine(net, initial_marking, final_marking, parameters)
        print(f'SIMULATION {i}...')
        if not start_ts_simulation:
            start_ts_simulation = log[0][0]['start:timestamp']
        sim_log = simulator_eng.apply(n_sim_traces, start_ts_simulation=start_ts_simulation)
        sim_logs.append(sim_log)

    return sim_logs, simulator_eng


def split_event_log(df_log, perc=0.8):

    map_new_cases = dict(zip(df_log['case:concept:name'].unique(), range(len(df_log['case:concept:name'].unique()))))
    df_log['case:concept:name'] = df_log['case:concept:name'].apply(lambda x: map_new_cases[x])
    df_train_log = df_log[df_log['case:concept:name'] < int(len(df_log['case:concept:name'].unique())*perc)]
    df_test_log = df_log[df_log['case:concept:name'] >= int(len(df_log['case:concept:name'].unique())*perc)]
    
    df_train_log['case:concept:name'] = df_train_log['case:concept:name'].astype(str)
    df_test_log['case:concept:name'] = df_test_log['case:concept:name'].astype(str)

    return df_train_log, df_test_log


def preprocessing_log(df_log):

    df_log['case:concept:name'] = df_log['case:concept:name'].astype(str)
    df_log['time:timestamp'] = pd.to_datetime(df_log['time:timestamp'])
    df_log['start:timestamp'] = pd.to_datetime(df_log['start:timestamp'])

    df_log.sort_values(by='start:timestamp', inplace=True)
    df_log.index = range(len(df_log))

    return df_log


def create_experiment_folder(
        df_train_log, 
        df_test_log,
        simulator_eng,
        df_sim_logs, 
        evaluations, 
        output_path
        ):
    
    os.mkdir(output_path)

    df_train_log.to_csv(output_path+'/df_train.csv', index=False)
    df_test_log.to_csv(output_path+'/df_test.csv', index=False)

    pm4py.write_pnml(simulator_eng.net, simulator_eng.initial_marking, simulator_eng.final_marking, output_path+'/petri_net.pnml')
    for i in range(len(df_sim_logs)):
        df_sim_logs[i].to_csv(output_path+f'/df_sim_{i+1}.csv', index=False)
        pd.DataFrame({'METRIC': evaluations.keys(), 'VALUES': [evaluations[k][i] for k in evaluations.keys()]}).to_csv(output_path+f'/evaluation_results_{i+1}.csv', index=False)
    pd.DataFrame({'METRIC': evaluations.keys(), 'VALUES': [np.mean(evaluations[k]) for k in evaluations.keys()]}).to_csv(output_path+f'/evaluation_results_avg.csv', index=False)
    pd.DataFrame({'METRIC': evaluations.keys(), 'VALUES': [np.std(evaluations[k]) for k in evaluations.keys()]}).to_csv(output_path+f'/evaluation_results_std.csv', index=False)