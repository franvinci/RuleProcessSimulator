import random
from tqdm import tqdm
from discovery.cf_discovery import discover_weight_transitions
from discovery.time_discovery import discover_execution_time_distributions, discover_arrival_time, discover_waiting_time
from discovery.calendar_discovery import discover_res_calendars, discover_arrival_calendar
from discovery.resource_discovery import discover_resource_acts_prob
from utils.common_utils import return_enabled_transitions, update_markings, return_fired_transition, compute_transition_weights_from_model, add_minutes_with_calendar
import pm4py
import datetime
import pandas as pd

import multiprocessing as mp



class SimulatorParameters:

    def __init__(self, net, initial_marking, final_marking):
        
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.net_transition_labels = list(set([t.label for t in net.transitions if t.label]))

        self.mode_history_weights = None
        self.label_data_attributes = []
        self.label_data_attributes_categorical = []
        self.attribute_values_label_categorical = dict()

        self.transition_weights = {t: 1 for t in list(self.net.transitions)}
        self.resources = ['res']
        self.act_resource_prob = {'res': {act: 1 for act in self.net_transition_labels}}
        self.calendars = {'res': {wd: {h: True for h in range(24)} for wd in range(7)}}
        self.arrival_calendar = {wd: {h: True for h in range(24)} for wd in range(7)}

        self.execution_time_distributions = {a: ('fixed', 1) for a in self.net_transition_labels}
        self.arrival_time_distributions = ('fixed', 1)
        self.waiting_time_distributions = {'res': ('fixed', 1)}

        self.max_ex_time = {a: 1 for a in self.net_transition_labels}
        self.max_ar_time = 1
        self.max_wt_time = 1


    def discover_from_eventlog(self, 
                               log, 
                               max_depths_cv=range(1, 6),
                               label_data_attributes=[], label_data_attributes_categorical=[], mode_history_weights=True):
        
        self.mode_history_weights = mode_history_weights
        self.label_data_attributes = label_data_attributes
        self.label_data_attributes_categorical = label_data_attributes_categorical
        
        for a in label_data_attributes_categorical:
            self.attribute_values_label_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

        self.transition_weights = discover_weight_transitions(
                                                                log, 
                                                                self.net, self.initial_marking, self.final_marking, 
                                                                self.net_transition_labels, 
                                                                max_depths_cv=max_depths_cv,                  
                                                                label_data_attributes=label_data_attributes, label_data_attributes_categorical=label_data_attributes_categorical, values_categorical=self.attribute_values_label_categorical,
                                                                mode_history_weights=mode_history_weights
                                                            )

        if label_data_attributes:
            self.distribution_data_attributes = []
            for trace in log:
                try:
                    self.distribution_data_attributes.append([trace[a] for a in label_data_attributes])
                except:
                    self.distribution_data_attributes.append([trace[0][a] for a in label_data_attributes])

        print("Calendars discovery...")
        self.calendars = discover_res_calendars(log)
        self.arrival_calendar = discover_arrival_calendar(log)

        print("Resources discovery...")
        self.resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
        self.act_resource_prob = discover_resource_acts_prob(log)


        self.execution_time_distributions, self.max_ex_time = discover_execution_time_distributions(
                                                                                                    log, self.net_transition_labels, self.calendars, 
                                                                                                    max_depths=max_depths_cv,
                                                                                                    label_data_attributes=label_data_attributes, label_data_attributes_categorical=label_data_attributes_categorical, values_categorical=self.attribute_values_label_categorical,
                                                                                                    mode_history_weights=mode_history_weights
                                                                                                    )
        
        self.arrival_time_distribution, self.max_ar_time = discover_arrival_time(log, self.arrival_calendar, max_depths=max_depths_cv)

        self.waiting_time_distributions, self.max_wt_time = discover_waiting_time(
                                                                                    log, 
                                                                                    self.calendars, 
                                                                                    label_data_attributes, label_data_attributes_categorical, self.attribute_values_label_categorical, 
                                                                                    max_depths=max_depths_cv
                                                                                )        



class SimulatorEngine:

    def __init__(self, net, initial_marking, final_marking, simulation_parameters: SimulatorParameters):

        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.simulation_parameters = simulation_parameters
        self.case_id = 1
        self.current_timestamp = None


    def generate_activities(self, x_attr=[]):

        trace = []
        trace_attributes = dict()
        for i, l in enumerate(self.simulation_parameters.label_data_attributes):
            trace_attributes[l] = x_attr[i]

        tkns = list(self.initial_marking)
        enabled_transitions = return_enabled_transitions(self.net, tkns)

        if self.simulation_parameters.mode_history_weights:
            x_history = {t_l: 0 for t_l in self.simulation_parameters.net_transition_labels}
            X = x_attr + list(x_history.values())
        else:
            X = x_attr

        if not X:
            transition_weights = self.simulation_parameters.transition_weights
        else:
            dict_x = dict(zip(self.simulation_parameters.label_data_attributes + self.simulation_parameters.net_transition_labels, X))
            for a in self.simulation_parameters.label_data_attributes_categorical:
                for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                    dict_x[a+' = '+str(v)] = (dict_x[a] == v)*1
                del dict_x[a]
            transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, dict_x)
        
        t_fired = return_fired_transition(transition_weights, enabled_transitions)
        if t_fired.label:
            trace.append(t_fired.label)

        tkns = update_markings(tkns, t_fired)
        while set(tkns) != set(self.final_marking):
            if t_fired.label:
                if self.simulation_parameters.mode_history_weights:
                    dict_x[t_fired.label] = 1
            transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, dict_x)
            enabled_transitions = return_enabled_transitions(self.net, tkns)
            t_fired = return_fired_transition(transition_weights, enabled_transitions)
            if t_fired.label:
                trace.append(t_fired.label)
            tkns = update_markings(tkns, t_fired)

        return trace, trace_attributes
        
    # Helper function for multiprocessing to generate traces
    def mp_helper_generate_trace_activities(self, i):
        if self.simulation_parameters.label_data_attributes:
            x_attr = random.sample(self.simulation_parameters.distribution_data_attributes, k=1)[0]
        else:
            x_attr = []
        return self.generate_activities(x_attr)


    def generate_events(self, curr_traces_acts, start_ts_simulation, curr_trace_attributes=[]):

        n_sim = len(curr_traces_acts)

        current_arr_ts = start_ts_simulation

        arrival_timestamps = dict()
        for id in range(self.case_id, self.case_id+n_sim):
            arrival_timestamps[f'case_{id}'] = current_arr_ts.timestamp()
            arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday(), 'month': current_arr_ts.month})
            current_arr_ts = add_minutes_with_calendar(current_arr_ts, int(arrival_delta), self.simulation_parameters.arrival_calendar)

        flag_attive_cases = {f'case_{id}': False for id in range(self.case_id, self.case_id+n_sim)}

        if self.simulation_parameters.mode_history_weights:
            hystory_active_traces = {f'case_{id}': {l: 0 for l in self.simulation_parameters.net_transition_labels} for id in range(self.case_id, self.case_id+n_sim)}
        events = []

        curr_trace_attribute_features = dict()
        if curr_trace_attributes:
            for case_id in curr_trace_attributes.keys():
                curr_trace_attribute_features[case_id] = dict()
                for a in self.simulation_parameters.label_data_attributes:
                    if a in self.simulation_parameters.label_data_attributes_categorical:
                        for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                            curr_trace_attribute_features[case_id][a+' = '+str(v)] = (curr_trace_attributes[case_id][a] == v)*1
                    else:
                        curr_trace_attribute_features[case_id][a] = curr_trace_attributes[case_id][a]

        while len(curr_traces_acts) > 0:

            curr_case_id = min(arrival_timestamps, key=arrival_timestamps.get)
            current_ts = arrival_timestamps[curr_case_id]

            if len(curr_traces_acts[curr_case_id]) < 1:
                del curr_traces_acts[curr_case_id]
                if self.simulation_parameters.mode_history_weights:
                    del hystory_active_traces[curr_case_id]
                del arrival_timestamps[curr_case_id]
                del curr_trace_attribute_features[curr_case_id]
                del curr_trace_attributes[curr_case_id]
                continue

            curr_act = curr_traces_acts[curr_case_id].pop(0)

            curr_res = random.choices(list(self.simulation_parameters.act_resource_prob[curr_act].keys()), weights=list(self.simulation_parameters.act_resource_prob[curr_act].values()))[0]

            n_active = 0
            if len(events) > 0:
                for e in events:
                    if e['org:resource'] != curr_res:
                        continue
                    if e['time:timestamp'] > current_ts and e['start:timestamp'] < current_ts:
                        n_active += 1

            current_ts_datetime = datetime.datetime.fromtimestamp(current_ts)

            if not flag_attive_cases[curr_case_id]:
                waiting_time = 0
                flag_attive_cases[curr_case_id] = True
            else:
                n_running_events = n_active
                if curr_res in self.simulation_parameters.waiting_time_distributions.keys():
                    if sum(hystory_active_traces[curr_case_id].values()) == 0:
                        waiting_time = 0
                    else:
                        waiting_time = self.simulation_parameters.waiting_time_distributions[curr_res].apply_distribution({'hour': current_ts_datetime.hour, 'weekday': current_ts_datetime.weekday(), 'month': current_ts_datetime.month, 'n. running events': n_running_events} | curr_trace_attribute_features[curr_case_id])
                else:
                    waiting_time = 0

            start_ts_datetime = add_minutes_with_calendar(current_ts_datetime, int(waiting_time), self.simulation_parameters.calendars[curr_res])
            start_ts = start_ts_datetime.timestamp()


            if self.simulation_parameters.mode_history_weights:
                ex_time = self.simulation_parameters.execution_time_distributions[curr_act].apply_distribution({'resource = '+res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | hystory_active_traces[curr_case_id] | curr_trace_attribute_features[curr_case_id] | {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday(), 'month': start_ts_datetime.month})
            else:
                ex_time = self.simulation_parameters.execution_time_distributions[curr_act].apply_distribution({res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | curr_trace_attribute_features[curr_case_id] | {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday(), 'month': start_ts_datetime.month})
            
            end_ts_datetime = add_minutes_with_calendar(start_ts_datetime, int(ex_time), self.simulation_parameters.calendars[curr_res])
            end_ts = end_ts_datetime.timestamp()
            arrival_timestamps[curr_case_id] = end_ts


            events.append({'case:concept:name': curr_case_id, 'concept:name': curr_act, 'start:timestamp': start_ts, 'time:timestamp': end_ts, 'org:resource': curr_res} | {a: curr_trace_attributes[curr_case_id][a] for a in self.simulation_parameters.label_data_attributes})

            if self.simulation_parameters.mode_history_weights:
                hystory_active_traces[curr_case_id][curr_act] += 1
            if len(curr_traces_acts[curr_case_id]) < 1:
                del curr_traces_acts[curr_case_id]
                if self.simulation_parameters.mode_history_weights:
                    del hystory_active_traces[curr_case_id]
                del arrival_timestamps[curr_case_id]
                del curr_trace_attribute_features[curr_case_id]
                del curr_trace_attributes[curr_case_id]

        df_events_sim = pd.DataFrame(events)
        df_events_sim['start:timestamp'] = df_events_sim['start:timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        df_events_sim['time:timestamp'] = df_events_sim['time:timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))

        self.current_timestamp = datetime.datetime.fromtimestamp(end_ts)

        return df_events_sim


    def apply_trace(self, start_ts_simulation=None, x_attr=[]):

        trace_acts, trace_attributes = self.generate_activities(x_attr)
        curr_traces_acts = {f'case_{self.case_id}': trace_acts}
        if start_ts_simulation:
            self.current_timestamp = start_ts_simulation
        sim_trace_df = self.generate_events(curr_traces_acts, self.current_timestamp, {f'case_{self.case_id}': trace_attributes})

        return sim_trace_df


    def apply(self, n_sim, start_ts_simulation):

        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            traces_acts_attributes = list(tqdm(pool.imap(self.mp_helper_generate_trace_activities, range(n_sim)), total=n_sim))

        curr_traces_acts = dict()
        curr_trace_attributes = dict()
        for j, id in enumerate(range(self.case_id, self.case_id+n_sim)):
            curr_traces_acts[f'case_{id}'] = traces_acts_attributes[j][0]
            curr_trace_attributes[f'case_{id}'] = traces_acts_attributes[j][1]

        sim_log_df = self.generate_events(curr_traces_acts, start_ts_simulation, curr_trace_attributes)

        self.case_id += n_sim

        return sim_log_df
