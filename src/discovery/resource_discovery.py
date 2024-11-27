import pm4py

def discover_resource_acts_prob(log):

    act_freq = pm4py.get_event_attribute_values(log, 'concept:name')

    act_resources = dict()
    for trace in log:
        for event in trace:
            res = event['org:resource']
            act = event['concept:name']
            if act not in act_resources.keys():
                act_resources[act] = dict()
            if res not in act_resources[act].keys():
                act_resources[act][res] = 1
            else:
                act_resources[act][res] += 1

    for act in act_resources.keys():
        for res in act_resources[act].keys():
            act_resources[act][res] = act_resources[act][res]/act_freq[act]

    return act_resources