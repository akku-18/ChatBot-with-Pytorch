import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?', ',', '!', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(all_words)

X_trains = []
y_train = []

for (patterns_sentence, tags) in xy:
    bag = bag_of_words(patterns_sentence, all_words)
    X_trains.append(bag)

    label = tags.index(tag)
    y_train.append(label) 

X_trains = np.array(X_trains)
y_train = np.array(y_train)



