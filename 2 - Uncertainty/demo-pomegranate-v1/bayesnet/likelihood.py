from pomegranate.bayesian_network import *

from model import model

# Calculate probability for a given observation ["none", "no", "on time", "attend"]
observation = [0, 1, 0, 0]
probability = model.probability([observation])
print(probability)