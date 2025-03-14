from pomegranate.distributions import Categorical
from pomegranate.markov_chain import *
from pomegranate.hmm import DenseHMM

import numpy

# ------------------------------------Hidden Markov Models------------------------------------

# Observation model for each state
probs_sun = torch.tensor([[0.2,     # umbrella
                            0.8]    # no umbrella
])
sun = Categorical(probs=probs_sun)

probs_rain = torch.tensor([[0.9,    # umbrella
                            0.1]    # no umbrella
])
rain = Categorical(probs=probs_rain)

states = [sun, rain]

# Transition model (prediction for tomorrow's weather)
edges = [
    [0.8, 0.2], # "sun": sun, rain
    [0.3, 0.7] # "rain": sun, rain
]

# Starting probabilities
starts = [0.5, 0.5]

# Create the model
model = DenseHMM(states, edges=edges, starts=starts)
