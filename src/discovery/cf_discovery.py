from utils.common_utils import return_transitions_frequency, return_enabled_and_fired_transitions
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from utils.dt_utils import DecisionRules
from sklearn.model_selection import GridSearchCV



def discover_weight_transitions(
        log, net, initial_marking, final_marking, 
        net_transition_labels, 
        max_depths_cv=range(1, 6),
        mode_transition_weights_ml=True, 
        label_data_attributes=[], label_data_attributes_categorical = [], values_categorical=None,
        mode_history_weights=True, transition_model_type='DecisionTree'
    ):

    print("Transition probabilities discovery...")

    if not mode_transition_weights_ml:
        transition_weights = return_transitions_frequency(log, net, initial_marking, final_marking)
        return transition_weights

    else:

        transition_weights = build_models(
                                            log, 
                                            net, 
                                            initial_marking, 
                                            final_marking,
                                            net_transition_labels,
                                            label_data_attributes,
                                            label_data_attributes_categorical,
                                            values_categorical,
                                            mode_history_weights,
                                            model_type=transition_model_type,
                                            max_depths_cv=max_depths_cv
                                        )
    
        return transition_weights
    

def build_models(
        log,
        net, 
        initial_marking, 
        final_marking,
        net_transition_labels,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        mode_history_weights,
        model_type='DecisionTree', 
        max_depths_cv=range(1,6)
    ):
    
    datasets_t = build_training_datasets(
                    log, net, initial_marking, final_marking, 
                    net_transition_labels, mode_history_weights, label_data_attributes
                )

    param_grid = {'max_depth': max_depths_cv}

    models_t = dict()
    
    for t in tqdm(net.transitions):
        data_t = datasets_t[t]
        if len(data_t['class'].unique())<2:
            models_t[t] = None
            continue
        
        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_t[a+' = '+str(v)] = (data_t[a] == v)*1
            del data_t[a]

        X = data_t.drop(columns=['class'])
        y = data_t['class']

        if model_type == 'LogisticRegression':
            clf_t = LogisticRegression(random_state=72).fit(X, y)
        elif model_type == 'DecisionTree':

            clf_t_dtc = DecisionTreeClassifier(random_state=72)
            try:
                grid_search = GridSearchCV(estimator=clf_t_dtc, param_grid=param_grid, cv=3).fit(X, y)
                clf_t_dtc = grid_search.best_estimator_
            except:
                clf_t_dtc = DecisionTreeClassifier(max_depth=2, random_state=72)
                clf_t_dtc.fit(X, y)

            clf_t = DecisionRules()
            clf_t.from_decision_tree(clf_t_dtc)

        models_t[t] = clf_t
    
    return models_t



def build_training_datasets(
        log, net, initial_marking, final_marking, 
        net_transition_labels, mode_history_weights, label_data_attributes
    ):

    if mode_history_weights:
        t_dicts_dataset = {t: {a: [] for a in label_data_attributes} | {t_l: [] for t_l in net_transition_labels} | {'class': []} for t in net.transitions}
    else:
        t_dicts_dataset = {t: {a: [] for a in label_data_attributes} | {'class': []} for t in net.transitions}

    alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]
    i = 0

    for trace in log:

        if label_data_attributes:
            try:
                trace_attributes = {a: trace[a] for a in label_data_attributes}
            except:
                trace_attributes = {a: trace[0][a] for a in label_data_attributes}

        trace_aligned = aligned_traces[i]
        i += 1
        visited_transitions, is_fired = return_enabled_and_fired_transitions(net, initial_marking, final_marking, trace_aligned)
        for j in range(len(visited_transitions)):
            t = visited_transitions[j]
            t_fired = is_fired[j]
            for a in label_data_attributes:
                t_dicts_dataset[t][a].append(trace_attributes[a])
            if mode_history_weights:
                transitions_fired = [label for label, value in zip(visited_transitions[:j], is_fired[:j]) if value == 1]
                for t_ in net_transition_labels:
                    # if history_weights == 'binary':
                    t_dicts_dataset[t][t_].append((t_ in [x.label for x in transitions_fired])*1)
                    # if history_weights == 'count':
                    #     t_dicts_dataset[t][t_].append([x.label for x in transitions_fired].count(t_))
            t_dicts_dataset[t]['class'].append(t_fired)
        
    datasets_t = {t: pd.DataFrame(t_dicts_dataset[t]) for t in net.transitions}

    return datasets_t
