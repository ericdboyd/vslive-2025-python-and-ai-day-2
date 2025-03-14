from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.markov_chain import *

# ------------------------------------Markov Model------------------------------------

# Define starting probabilities
probs_start = torch.tensor([[0.5,   # sun
                            0.5]    # rain
])
start = Categorical(probs=probs_start)

# Define transition model
probs_transitions = torch.tensor([
    [0.8, 0.2],  # sun [start]: sun, rain
    [0.3, 0.7]   # rain [start]: sun, rain
])
transitions = ConditionalCategorical(probs=[probs_transitions])

# Create Markov chain
model = MarkovChain([start, transitions])

# Sample 50 states from chain (if starts from sun, can change if desired)
sample = []
for i in range(100):
    samples = model.sample(1)
    if samples[:, 0] == 0: 
        sample.append(samples[:, 1].item())
print(sample)