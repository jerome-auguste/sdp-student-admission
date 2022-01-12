"""Standardised evaluation on models"""

from ncs import NcsSatModel
from generator import Generator
from collections import Counter


gen = Generator(size=400, num_classes=3, num_criterions=5, lmbda=0.6)

# for _ in range(100):
train_set, labels = gen.generate()
while set(labels) != set(range(gen.num_classes)):
    # Checks if at least one datapoint belongs to each class
    train_set, labels = gen.generate()


print(f"Parameters:\n",
            f"- lambda: {gen.lmbda}\n",
            f"- weights: {gen.weights}\n",
            f"- frontier: {gen.frontier}\n",
            f"- echantillons: {dict(Counter(labels))}\n"
        )

model = NcsSatModel(generator=gen, train_set=train_set, labels=labels)

res = model.solve()

# print(res)