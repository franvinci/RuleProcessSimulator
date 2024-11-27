import sys
sys.path.append('src/')

from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from experiment_utils import discovery_and_simulate, split_event_log, preprocessing_log, create_experiment_folder
from evaluation import evaluate
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

CASE_STUDIES = {
    'purchasing': {
        'PATH_LOG': 'data/logs/purchasing.xes',
        'PATH_MODEL': 'data/models/purchasing.pnml',
        'LABEL_ATTR': [],
        'LABEL_ATTR_CATEGORICAL': [],
        'OUTPUT_PATH': 'outputs_prova'
    },
    'acr': {
        'PATH_LOG': 'data/logs/acr.xes',
        'PATH_MODEL': 'data/models/acr.pnml',
        'LABEL_ATTR': [],
        'LABEL_ATTR_CATEGORICAL': [],
        'OUTPUT_PATH': 'outputs'
    },
    'cvs': {
        'PATH_LOG': 'data/logs/cvs.xes',
        'PATH_MODEL': 'data/models/cvs.pnml',
        'LABEL_ATTR': [],
        'LABEL_ATTR_CATEGORICAL': [],
        'OUTPUT_PATH': 'outputs'
    },
    'bpi12': {
        'PATH_LOG': 'data/logs/bpi12.xes',
        'PATH_MODEL': 'data/models/bpi12.pnml',
        'LABEL_ATTR': ['AMOUNT_REQ'],
        'LABEL_ATTR_CATEGORICAL': [],
        'OUTPUT_PATH': 'outputs'
    },
    'bpi17': {
        'PATH_LOG': 'data/logs/bpi17.xes',
        'PATH_MODEL': 'data/models/bpi17.pnml',
        'LABEL_ATTR': ['LoanGoal', 'ApplicationType', 'RequestedAmount'],
        'LABEL_ATTR_CATEGORICAL': ['LoanGoal', 'ApplicationType'],
        'OUTPUT_PATH': 'outputs'
    }
}

N_SIM = 5
MAX_DEPTHS = range(1,6)


if __name__ == "__main__":

    for case_study in CASE_STUDIES.keys():

        print(f'\nRun Experiments for case study: {case_study}')

        log = xes_importer.apply(CASE_STUDIES[case_study]['PATH_LOG'])

        print('PREPROCESSING...')
        df_log = pm4py.convert_to_dataframe(log)
        df_log = preprocessing_log(df_log)
        print('SPLIT TRAIN-TEST...')
        df_train_log, df_test_log = split_event_log(df_log)

        train_log = pm4py.convert_to_event_log(df_train_log)

        n_sim_traces = len(df_test_log['case:concept:name'].unique())

        net, initial_marking, final_marking = pm4py.read_pnml(CASE_STUDIES[case_study]['PATH_MODEL'])

        df_sim_logs, simulator_eng = discovery_and_simulate(
            train_log,
            net, initial_marking, final_marking,
            max_depths=MAX_DEPTHS,
            noise_threshold_im=0.0,
            start_ts_simulation=df_test_log.iloc[0]['start:timestamp'],
            label_data_attributes=CASE_STUDIES[case_study]['LABEL_ATTR'], 
            label_data_attributes_categorical=CASE_STUDIES[case_study]['LABEL_ATTR_CATEGORICAL'],
            n_sim_traces=n_sim_traces,
            n_sim=N_SIM
        )

        print('EVALUATION...')

        evaluations = dict()
        for i in range(N_SIM):
            metrics = evaluate(df_test_log, df_sim_logs[i])
            for metric in metrics.keys():
                if i == 0:
                    evaluations[metric] = [metrics[metric]]
                else:
                    evaluations[metric].append(metrics[metric])

        for metric in evaluations.keys():
            print(metric, ': ', np.mean(evaluations[metric]))

        os.mkdir(CASE_STUDIES[case_study]['OUTPUT_PATH']) if CASE_STUDIES[case_study]['OUTPUT_PATH'] not in os.listdir() else None

        create_experiment_folder(
            df_train_log, 
            df_test_log,
            simulator_eng,
            df_sim_logs,
            evaluations, 
            CASE_STUDIES[case_study]['OUTPUT_PATH'] + '/' + case_study
        )

