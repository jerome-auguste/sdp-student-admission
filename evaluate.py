"""Standardised evaluation on models"""

from ncs import NcsSatModel
from generator import Generator


gen = Generator(size=50, num_classes=5, num_criterions=5)
train_set, labels = gen.generate()
while set(labels) != set(range(gen.num_classes)):
    # Checks if at least one datapoint belongs to each class
    train_set, labels = gen.generate()

model = NcsSatModel(generator=gen, train_set=train_set, labels=labels)

res = model.solve()

print(res)