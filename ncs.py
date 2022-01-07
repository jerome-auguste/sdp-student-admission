import numpy as np
from itertools import product
from pysat.solvers import Glucose3
from generator import Generator
from collections import Counter

gen = Generator(size=10)

grades,admission = gen.generate()
print(f"Parameters:\nlambda: {gen.lmbda}\nweights: {gen.weights}\nfrontier: {gen.frontier}\nelements: {dict(Counter(admission))}\n")

rate = 0
num_iter = 1000
mean = 0
std = 0

for i in range(num_iter):
    try:
        gen = Generator(size=10)
        grades,admission = gen.generate()
        rate += dict(Counter(admission))[1.0]/gen.size
        mean += np.mean(grades)
        std += np.std(grades)
    except KeyError:
        pass

print(f"Mean of 1 rate: {rate/num_iter}")
print(f"Mean grade: {mean/num_iter}")
print(f"Std grade: {std/num_iter}")
# g = Glucose3()
# g.add_clause([-1, 2])
# g.add_clause([-2, 3])
# print(g.solve())
# print(g.get_model())

