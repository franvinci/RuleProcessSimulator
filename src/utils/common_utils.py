from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.dt_utils import DecisionRules
from datetime import timedelta



def return_transitions_frequency(log, net, initial_marking, final_marking):

    alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]

    frequency_t = {t: 0 for t in net.transitions}
    for trace in aligned_traces:
        for align in trace:
            name_t = align[1]
            for t in list(net.transitions):
                if t.name == name_t:
                    frequency_t[t] += 1
                    break

    return frequency_t


def return_enabled_and_fired_transitions(net, initial_marking, final_marking, trace_aligned):

    visited_transitions = []
    is_fired = []
    tkns = list(initial_marking)
    enabled_transitions = return_enabled_transitions(net, tkns)
    for t_fired_name in trace_aligned:
        for t in net.transitions:
            if t.name == t_fired_name[1]:
                t_fired = t
                break
        not_fired_transitions = list(enabled_transitions-{t_fired})
        for t_not_fired in not_fired_transitions:
            visited_transitions.append(t_not_fired)
            is_fired.append(0)
        visited_transitions.append(t_fired)
        is_fired.append(1)
        tkns = update_markings(tkns, t_fired)
        if set(tkns) == set(final_marking):
            return visited_transitions, is_fired
        enabled_transitions = return_enabled_transitions(net, tkns)

    return visited_transitions, is_fired


def update_markings(tkns, t_fired):

    for a_in in list(t_fired.in_arcs):
        tkns.remove(a_in.source)
    for a_out in list(t_fired.out_arcs):
        tkns.extend([a_out.target])        
    
    return tkns


def return_enabled_transitions(net, tkns):
    
    enabled_t = set()
    for t in list(net.transitions):
        if {a.source for a in t.in_arcs}.issubset(tkns):
            enabled_t.add(t)
    
    return enabled_t


def return_fired_transition(transition_weights, enabled_transitions):

    total_weight = sum(transition_weights[s] for s in enabled_transitions)
    random_value = random.uniform(0, total_weight)
    
    cumulative_weight = 0
    for s in enabled_transitions:
        cumulative_weight += transition_weights[s]
        if random_value <= cumulative_weight:
            return s
        

def compute_transition_weights_from_model(models_t, dict_x):
    transition_weights = dict()
    for t in list(models_t.keys()):
        if type(models_t[t]) in [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
            X = pd.DataFrame({k: [dict_x[k]] for k in dict_x.keys()})
            transition_weights[t] = compute_proba(models_t, t, X)
        elif type(models_t[t]) == DecisionRules:
            transition_weights[t] = models_t[t].apply(dict_x)
        else:
            transition_weights[t] = 1
    return transition_weights


def compute_proba(models_t, t, X):
    
    clf_t = models_t[t]
    
    return clf_t.predict_proba(X)[0,1]


def count_false_hours(calendar, start_ts, end_ts):
    false_hours_count = 0
    current_time = start_ts
    
    while current_time < end_ts:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if calendar.get(weekday, {}).get(hour) == False:
            false_hours_count += 1
            
        current_time += timedelta(hours=1)

    return false_hours_count


def add_minutes_with_calendar(start_ts, minutes_to_add, calendar):
    remaining_minutes = minutes_to_add
    current_time = start_ts

    while remaining_minutes > 0:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if calendar.get(weekday, {}).get(hour, False):
            minutes_in_current_hour = min(remaining_minutes, 60 - current_time.minute)
            
            current_time += timedelta(minutes=minutes_in_current_hour)
            remaining_minutes -= minutes_in_current_hour
        else:
            current_time = (current_time + timedelta(hours=1)).replace(minute=0)

    return current_time