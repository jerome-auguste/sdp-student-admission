import numpy as np
from itertools import product

from numpy.core.fromnumeric import size
from pysat.solvers import Glucose3
from generator import Generator
from collections import Counter


train_set_size = 10
num_classes = 3
gen = Generator(size=train_set_size, num_classes=num_classes)

grades,admission = gen.generate()
print(f"Parameters:\nlambda: {gen.lmbda}\nweights: {gen.weights}\nfrontier: {gen.frontier}\nelements: {dict(Counter(admission))}\n")

print(admission)
# Boolean dimensions x_(i, h, k):
# - i is criterion (grades columns)
# - h is (ordered) class number (admission value)
# - k value taken for a criterion (grades value)

# y_B:
# - B coalition of criteria (B included in N)
class_dict = {}
for cat in range(gen.num_classes):
    class_dict[cat] = []
    for student in range (train_set_size):
        for crit in range(gen.num_criterions):
            if admission[student] == cat:
                class_dict[cat].append([student, crit])

print(class_dict)


g = Glucose3()


# 3a clause : all pairs of values k < k' for all i and h:



# rate = 0
# num_iter = 1000
# mean = 0
# std = 0

# for i in range(num_iter):
#     try:
#         gen = Generator(size=10)
#         grades,admission = gen.generate()
#         rate += dict(Counter(admission))[1.0]/gen.size
#         mean += np.mean(grades)
#         std += np.std(grades)
#     except KeyError:
#         pass

# print(f"Mean of 1 rate: {rate/num_iter}")
# print(f"Mean grade: {mean/num_iter}")
# print(f"Std grade: {std/num_iter}")


# g = Glucose3()
# g.add_clause([-1, 2])
# g.add_clause([-2, 3])
# print(g.solve())
# print(g.get_model())

