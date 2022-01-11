"""Standardised evaluation on models"""

from ncs import NcsSatModel
from generator import Generator


for _ in range(100):
    gen = Generator(size=50, num_classes=5, num_criterions=5)
    train_set, labels = gen.generate()
    while set(labels) != set(range(gen.num_classes)):
        # Checks if at least one datapoint belongs to each class
        train_set, labels = gen.generate()

    print(f"Labels: {set(labels)}")
    model = NcsSatModel(generator=gen, train_set=train_set, labels=labels)

    res = model.solve()

# print(res)