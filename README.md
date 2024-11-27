# Rule-Aware Process Simulator

A Python Package for building and discovering Rule-Aware Business Process Simulation from event log data.


### How to use:

<ol>
    <li>
        <strong>Clone this repository.</strong>
    </li>
    <li>
        <strong>Create environment:</strong>
        <pre><code>$ conda env create -f environment.yml</code></pre>
    </li>
</ol>


```python
import simulator
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

# read input Event log and Petri net
log = xes_importer.apply('data/logs/purchasing.xes')
net, im, fm = pm4py.read_pnml("data/models/purchasing.pnml")

# initialize simulation parameters
parameters = simulator.SimulatorParameters(net, initial_marking, final_marking)
# discover from event log
parameters.discover_from_eventlog(log)

# initialize simulation engine
simulator_eng = simulator.SimulatorEngine(net, initial_marking, final_marking, parameters)
# simulate event log
sim_log = simulator_eng.apply(n_sim_traces, start_ts_simulation=log[0][0]['start:timestamp'])
```

### Experiments Replicability Instructions:

Input files are in <a href="data.zip"><code>data.zip</code></a>.
Output files are in <a href="output.zip"><code>output.zip</code></a>.


#### To re-launch experiments:
<ol>
    <li>
        <strong>Unzip <code>data.zip</code>:</strong>
        <pre><code>$ unzip data.zip</code></pre>
    </li>
    <li>
        <strong>Run experiments:</strong>
        <pre><code>$ python main.py</code></pre>
    </li>
</ol>