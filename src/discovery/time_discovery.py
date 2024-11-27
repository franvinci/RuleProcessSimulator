import numpy as np
from utils.dt_utils import DecisionRules
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd
import pm4py
import datetime
from utils.common_utils import count_false_hours
from utils.distribution_utils import return_best_distribution, sampling_from_dist



# DISCOVERY

def discover_arrival_time(
        log, 
        calendar_arrival, 
        max_depths=range(1,6)
    ):

    print("Arrival Time discovery...")
    arrival_time_distr, max_arrival_time = build_model_arrival(log, calendar_arrival, max_depths)

    return arrival_time_distr, max_arrival_time



def discover_execution_time_distributions(
        log, 
        net_transition_labels, 
        calendars, 
        max_depths=range(1,6),
        label_data_attributes=[], label_data_attributes_categorical=[], values_categorical=None,
        mode_history_weights=True
    ):

    print("Execution Time discovery...")
    activity_exec_time_distributions, max_exec_time_act = build_models_ex(
                                                                            log, 
                                                                            net_transition_labels, 
                                                                            calendars,
                                                                            label_data_attributes, label_data_attributes_categorical, values_categorical,
                                                                            mode_history_weights, max_depths
                                                                        )

    return activity_exec_time_distributions, max_exec_time_act



def discover_waiting_time(
        log, 
        calendars, 
        label_data_attributes, label_data_attributes_categorical, values_categorical, 
        max_depths=range(1,6)
    ):

    print("Waiting Time discovery...")
    role_waiting_time_distributions, max_wt_role = build_models_wt(
                                                                    log, 
                                                                    calendars, 
                                                                    label_data_attributes, label_data_attributes_categorical, values_categorical, 
                                                                    max_depths
                                                                )

    return role_waiting_time_distributions, max_wt_role



# BUILD ML MODELS

def build_model_arrival(
        log, 
        calendar_arrival, 
        max_depths=range(1,6)
    ):

    param_grid = {'max_depth': max_depths}

    df = build_training_df_arrival(log, calendar_arrival)

    X = df.drop(columns=['arrival_time'])
    y = df['arrival_time']
    max_arrival_time = int(max(y))

    if len(X)>3:
        clf_mean = DecisionTreeRegressor(random_state=72)
        grid_search = GridSearchCV(estimator=clf_mean, param_grid=param_grid, cv=3).fit(X, y)
        clf_mean = grid_search.best_estimator_
    else:
        clf_mean = DecisionTreeRegressor(max_depth=1, random_state=72).fit(X, y)

    leaf_indices = clf_mean.apply(X)

    y_leaf = pd.DataFrame({
        'Leaf': leaf_indices,
        'Y': y
    })

    clf = DecisionRules()
    clf.from_decision_tree(clf_mean)

    leaves = list(y_leaf['Leaf'].unique())
    for l in leaves:
        y = y_leaf[y_leaf['Leaf']==l]['Y']
        dist, params = return_best_distribution(y)
        max_value = int(max(y))
        min_value = int(min(y))
        sampled = sampling_from_dist(dist, params, min_value, max_value, np.median(y))
        clf.rules[l]['dist'] = dist, params
        clf.rules[l]['sampled'] = list(sampled)

    return clf, max_arrival_time



def build_models_ex(
        log, 
        activity_labels, 
        calendars,
        label_data_attributes, label_data_attributes_categorical, values_categorical, 
        mode_history_weights, 
        max_depths=range(1,6)
    ):

    df = build_training_df_ex(
                                log, 
                                activity_labels, 
                                calendars,
                                label_data_attributes, label_data_attributes_categorical, values_categorical, 
                                mode_history_weights
                            )
    
    param_grid = {'max_depth': max_depths}

    models_act = dict()
    max_exec_time_act = dict()

    for act in tqdm(activity_labels):

        df_act = df[df['activity_executed'] == act].iloc[:,1:]

        X = df_act.drop(columns=['execution_time'])
        y = df_act['execution_time']
        max_exec_time_act[act] = int(max(y))

        if len(X)>3:
            clf_mean = DecisionTreeRegressor(random_state=72)
            grid_search = GridSearchCV(estimator=clf_mean, param_grid=param_grid, cv=3).fit(X, y)
            clf_mean = grid_search.best_estimator_
        else:
            clf_mean = DecisionTreeRegressor(max_depth=1, random_state=72).fit(X, y)

        leaf_indices = clf_mean.apply(X)

        y_leaf = pd.DataFrame({
            'Leaf': leaf_indices,
            'Y': y
        })

        clf = DecisionRules()
        clf.from_decision_tree(clf_mean)

        leaves = list(y_leaf['Leaf'].unique())
        for l in leaves:
            y = y_leaf[y_leaf['Leaf']==l]['Y']
            dist, params = return_best_distribution(y)
            max_value = int(max(y))
            min_value = int(min(y))
            sampled = sampling_from_dist(dist, params, min_value, max_value, np.median(y))
            clf.rules[l]['dist'] = dist, params
            clf.rules[l]['sampled'] = list(sampled)

        models_act[act] = clf

    return models_act, max_exec_time_act



def build_models_wt(
        log, 
        calendars, 
        label_data_attributes, label_data_attributes_categorical, values_categorical, 
        max_depths=range(1,6)
    ):

    resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
    df_per_res = build_training_df_wt(log, calendars, label_data_attributes)

    max_wt_res = dict()

    param_grid = {'max_depth': max_depths}

    models_res = dict()
    for res in tqdm(resources):
        df_res = df_per_res[res]
        if len(df_res) <1:
            continue

        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                df_res[a+' = '+str(v)] = (df_res[a] == v)*1
            del df_res[a]

        X = df_res.drop(columns=['waiting_time'])
        y = df_res['waiting_time']
        max_wt_res[res] = int(max(y))

        if len(X)>3:
            clf_mean = DecisionTreeRegressor(random_state=72)
            grid_search = GridSearchCV(estimator=clf_mean, param_grid=param_grid, cv=3).fit(X, y)
            clf_mean = grid_search.best_estimator_
        else:
            clf_mean = DecisionTreeRegressor(max_depth=1, random_state=72).fit(X, y)


        leaf_indices = clf_mean.apply(X)

        y_leaf = pd.DataFrame({
            'Leaf': leaf_indices,
            'Y': y
        })

        clf = DecisionRules()
        clf.from_decision_tree(clf_mean)

        leaves = list(y_leaf['Leaf'].unique())
        for l in leaves:
            y = y_leaf[y_leaf['Leaf']==l]['Y']
            dist, params = return_best_distribution(y)
            max_value = int(max(y))
            min_value = int(min(y))
            sampled = sampling_from_dist(dist, params, min_value, max_value, np.median(y))
            clf.rules[l]['dist'] = dist, params
            clf.rules[l]['sampled'] = list(sampled)
        
        models_res[res] = clf

    return models_res, max_wt_res



# BUILD TRAINING DATASETS

def build_training_df_arrival(
        log, 
        calendar_arrival, 
        START_TS_LABEL= 'start:timestamp'
    ):

    dict_df = {'hour': []} | {'weekday': []} | {'month': []} | {'arrival_time': []}

    for i in range(len(log)-1):
        trace_cur = log[i]
        trace_next = log[i+1]

        cur_ts = trace_cur[0][START_TS_LABEL]
        next_ts = trace_next[0][START_TS_LABEL]

        dict_df['hour'].append(cur_ts.hour)
        dict_df['weekday'].append(cur_ts.weekday())
        dict_df['month'].append(cur_ts.month)

        ar_time = max((next_ts - cur_ts).total_seconds()/60 - count_false_hours(calendar_arrival, cur_ts, next_ts)*60, 0)
        dict_df['arrival_time'].append(ar_time)   
    
    df = pd.DataFrame(dict_df)
    
    return df



def build_training_df_ex(
        log, 
        activity_labels, 
        calendars,
        label_data_attributes, label_data_attributes_categorical, values_categorical, 
        mode_history_weights,
        START_TS_LABEL='start:timestamp', END_TS_LABEL='time:timestamp'
    ):
    
    if mode_history_weights:
        dict_df = {'activity_executed': []} | {attr: [] for attr in label_data_attributes} | {'resource': []} | {act: [] for act in activity_labels} | {'hour': []} | {'weekday': []} | {'month': []} | {'execution_time': []}
    else:
        dict_df = {'activity_executed': []} | {attr: [] for attr in label_data_attributes} | {'resource': []} | {'hour': []} | {'weekday': []} | {'month': []} | {'execution_time': []}


    for trace in log:
        
        try:
            trace_attributes = {a: trace[a] for a in label_data_attributes}
        except:
            trace_attributes = {a: trace[0][a] for a in label_data_attributes}

        if mode_history_weights:
            trace_history = {a: 0 for a in activity_labels}
        
        for event in trace:
            if event['concept:name'] not in activity_labels:
                continue

            for a in label_data_attributes:
                dict_df[a].append(trace_attributes[a])
            
            if mode_history_weights:
                for a in activity_labels:
                    dict_df[a].append(trace_history[a])

            act_executed = event['concept:name']
            dict_df['activity_executed'].append(act_executed)
            if mode_history_weights:
                trace_history[act_executed] += 1

            start_ts = event[START_TS_LABEL]
            end_ts = event[END_TS_LABEL]

            res = event['org:resource']

            dict_df['resource'].append(res)

            dict_df['hour'].append(start_ts.hour)
            dict_df['weekday'].append(start_ts.weekday())
            dict_df['month'].append(start_ts.month)

            ex_time = max((end_ts - start_ts).total_seconds()/60 - count_false_hours(calendars[res], start_ts, end_ts)*60, 0)
            dict_df['execution_time'].append(ex_time)
        
    df = pd.DataFrame(dict_df)

    resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
    for r in resources:
        df['resource = '+r] = (df['resource'] == r)*1
    del df['resource']

    for a in label_data_attributes_categorical:
        for v in values_categorical[a]:
            df[a+' = '+str(v)] = (df[a] == v)*1
        del df[a]

    return df



def build_training_df_wt(
        log, 
        calendars, 
        label_data_attributes, 
        START_TS_LABEL='start:timestamp', END_TS_LABEL='time:timestamp'
    ):

    resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())

    df_log = pm4py.convert_to_dataframe(log)

    # TODO METTERE UN CHECK PER QUESTO
    df_log[START_TS_LABEL] = df_log[START_TS_LABEL].apply(lambda x: datetime.datetime.fromisoformat(str(x)[:-6]).timestamp())
    df_log[END_TS_LABEL] = df_log[END_TS_LABEL].apply(lambda x: datetime.datetime.fromisoformat(str(x)[:-6]).timestamp())

    dict_df_per_res = {res: {attr: [] for attr in label_data_attributes} | {'hour': [], 'weekday': [], 'month': [], 'n. running events': [], 'waiting_time': []} for res in resources}

    for trace in log:
        try:
            trace_attributes = {a: trace[a] for a in label_data_attributes}
        except:
            trace_attributes = {a: trace[0][a] for a in label_data_attributes}
        for i in range(1,len(trace)):
            res = trace[i]['org:resource']
            for attr in label_data_attributes:
                dict_df_per_res[res][attr].append(trace_attributes[attr])
            prev_ts = trace[i-1][END_TS_LABEL]
            start_ts = trace[i][START_TS_LABEL]
            dict_df_per_res[res]['hour'].append(prev_ts.hour)
            dict_df_per_res[res]['weekday'].append(prev_ts.weekday())
            dict_df_per_res[res]['month'].append(prev_ts.month)
            df_log_filtered = df_log[(df_log[END_TS_LABEL] > prev_ts.timestamp()) & (df_log[START_TS_LABEL] < prev_ts.timestamp())]
            n_active_ev = (df_log_filtered['org:resource']==res).sum()
            dict_df_per_res[res]['n. running events'].append(n_active_ev)
            dict_df_per_res[res]['waiting_time'].append(max((start_ts - prev_ts).total_seconds()/60 - count_false_hours(calendars[res], prev_ts, start_ts)*60 , 0))

    df_per_res = {res: pd.DataFrame(dict_df_per_res[res]) for res in resources}

    return df_per_res