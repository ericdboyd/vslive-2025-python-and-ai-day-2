from pomegranate.bayesian_network import *

from torch.masked import MaskedTensor

from model import model

# Calculate predictions based on the evidence that the train was delayed
observation = torch.tensor([[0, 0, 1, 0]])
mask = torch.tensor([[False, False, True, False]])
X = MaskedTensor(observation, mask)

probabilities = model.predict_proba(X)
print(probabilities)