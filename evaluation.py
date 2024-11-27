import pandas as pd

from log_distance_measures.config import EventLogIDs

from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance, discretize_to_hour
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance

import datetime



def evaluate(original_log, simulated_log, resources_av = True):

    event_log_ids = EventLogIDs(  
        case="case:concept:name",
        activity="concept:name",
        resource="org:resource",
        start_time="start:timestamp",
        end_time="time:timestamp"
    )


    original_log[event_log_ids.start_time] = pd.to_datetime(original_log[event_log_ids.start_time], format='ISO8601', utc=True)
    original_log[event_log_ids.end_time] = pd.to_datetime(original_log[event_log_ids.end_time], format='ISO8601', utc=True)

    simulated_log[event_log_ids.start_time] = pd.to_datetime(simulated_log[event_log_ids.start_time], utc=True)
    simulated_log[event_log_ids.end_time] = pd.to_datetime(simulated_log[event_log_ids.end_time], utc=True)

    metrics = dict()

    metrics['cfld'] = control_flow_log_distance(
        original_log, 
        event_log_ids, 
        simulated_log, 
        event_log_ids, 
        True
    )

    metrics['ngd'] = n_gram_distribution_distance(
        original_log, 
        event_log_ids, 
        simulated_log, 
        event_log_ids, 
        n=3,
    )

    metrics['aed'] = absolute_event_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids,
            AbsoluteTimestampType.BOTH,
            discretize_to_hour,
    )

    metrics['ced'] = circadian_event_distribution_distance(
        original_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        AbsoluteTimestampType.BOTH,
    )

    if resources_av:
        metrics['cwd'] = circadian_workforce_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids
        )
    else:
        metrics['cwd'] = 0

    metrics['red'] = relative_event_distribution_distance(
        original_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        AbsoluteTimestampType.BOTH,
    )

    metrics['car'] = case_arrival_distribution_distance(
        original_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
    )

    metrics['ctd'] = cycle_time_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids,
            datetime.timedelta(hours=1),
    )

    return metrics