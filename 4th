import pandas as pd
import numpy as np
dataset = pd.read_csv("PlayTennis.csv")
print(dataset)

data = []
for row in dataset.values:
    data.append(row)
data = np.array(data)
print(data)
features = list(dataset.columns[: -1])
print(features)

import math
def entropy(data):
    yes = 0
    no = 0
    for row in data:
       if row[-1] == 'yes':
           yes += 1
       else:
           no += 1
           
    if yes == 0 or no == 0:
        return 0
    total = yes + no
    p_yes = yes/total
    p_no = no/total
    return -(p_yes * math.log2(p_yes)) - (p_no * math.log2(p_no))

def Information_gain(data,column):
    total_entropy = entropy(data)
    weighted_entropy = 0
    values  = set([row[column] for row in data])
    for value in values :
        subset = []
        for row in data:
            if row[column] == value:
               subset.append(row)
        weighted_entropy += entropy(subset) * (len(subset)/len(data))
    return total_entropy - weighted_entropy

def best_attribute(data):
    gains = []
    for i in range(len(features)):
        gain = Information_gain(data,i)
        gains.append(gain)
    max_gain = max(gains)
    return features[gains.index(max_gain)]

print(entropy(data))
print("-------------------------")
for i in range(len(features)):
    print(features[i],"=",Information_gain(data,i))
print("---------------------------")
print(best_attribute(data))
