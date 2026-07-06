Orbit Wars Kaggle submission bundle
===================================

4-player checkpoint: iter_00003960.pt (copied to checkpoint_4p.pt)
2-player checkpoint: iter_00004740.pt (copied to checkpoint_2p.pt)
4-player search uses embedded student: True
2-player search uses embedded student: True
4-player search ego main-model steps: 10
2-player search ego main-model steps: 10
Greedy (default): False
Greedy 4p override: default
Greedy 2p override: default
Sampling mode (default): mixed
Sampling mode 4p override: mixed
Sampling mode 2p override: mixed
Device: cpu
CPU threads: 1
Population member (fallback): member 0 / checkpoint default
Population member 4p: member 0 / checkpoint default
Population member 2p: member 0 / checkpoint default
Target method: interval
Interval geometry: tangent
Model search steps: disabled / env default
Model search steps 4p override: default
Model search steps 2p override: default
Model search adaptive horizon: True
Model search adaptive horizon offset: 3
Model search min overage seconds: 10.0
Model search gamma: checkpoint/default
Model search launch probability threshold: 0.05
Model search greedy launch threshold: env default
Model search branch after first env step: env default
Model search stop at turn end: env default
Model search turn-end opponent samples: env default
Model search turn-sampling max samples: env default

Policy selection: 4-player matches use checkpoint_4p.pt; 2-player matches
use checkpoint_2p.pt. The first observation selects the mode for the full
episode; there is no mid-game policy switching. When enabled, search uses
the embedded student model from the corresponding checkpoint, optionally
keeping the main model on the ego seat for the first configured number of
simulated search env steps.

Test locally (from this directory):

    pip install "kaggle-environments>=1.28.0" torch numpy
    python -c "
from kaggle_environments import make
from main import agent
env = make('orbit_wars', configuration={'seed': 42, 'agentCount': 4}, debug=True)
env.run([agent] * 4)
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

Submit:

    kaggle competitions submit orbit-wars -f /home/billy/orbit-wars/dist/2p-4p.tar.gz -m "your message"
