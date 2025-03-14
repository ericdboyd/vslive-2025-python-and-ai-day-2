from pomegranate.bayesian_network import *

from collections import Counter

from model import model

# ------------------------------------Rejection Sampling------------------------------------

# Compute distribution of Appointment given that train is delayed
N = 10000
data = []

# Repeat sampling 10,000 times
for i in range(N):
    # Generate a sample based on the function that we defined earlier
    sample = model.sample(1)

    # If, in this sample, the variable of Train has the value delayed, save the sample.
    # Since we are interested in the probability distribution of Appointment given that the train is delayed,
    # we discard the samples where the train was on time.
    if sample[:, 2] == 1:  # Assuming train delayed is encoded as 1
        data.append(sample[:, 3].item())  # Appointment

# Count how many times each value of the variable appeared
count = Counter(data)
prob_attend = count[0] / sum(count.values())
print(count)
print(f"probability that you attend given train is on time: {prob_attend:.4f}")