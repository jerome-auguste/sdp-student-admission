#%%
from gurobipy import *
import numpy as np
from numpy.core.fromnumeric import shape
from generator import Generator
from itertools import product
from collections import Counter
import sys
np.set_printoptions(precision=2)

gen = Generator(size=1000)
grades,admission = gen.generate()
print(f"Parameters:\nlambda: {gen.lmbda}\nweights: {gen.weights}\nfrontier: {gen.frontier}\nelements: {dict(Counter(admission))}\n")

model = Model("MR-sort")

# Constants
nb_ech = gen.size
nb_notes = gen.num_criterions
epsilon = 1e-9
M = 1e2 # superieur a l'ecart max, 20

# Gurobi variables
alpha = model.addVar()
x = model.addMVar(shape=nb_ech) # slack for each student (in A*)
y = model.addMVar(shape=nb_ech) # slack for each student (in R*)
w = model.addMVar(shape=nb_notes, lb=0, ub=1)
b = model.addMVar(shape=nb_notes)

lmbda = model.addVar(lb=0.5, ub=1)

c = model.addMVar(shape=(nb_ech, nb_notes), lb=0, ub=1)
d = model.addMVar(shape=(nb_ech, nb_notes), vtype=GRB.BINARY)

# Constraints
# Notes :
# - les contraintes doivent etre des 1D MVars objects
# - la multiplication element par element (a*b) semble
#   poser probleme avec les variables Gurobi

rejected = [j for j in range(nb_ech) if not admission[j]]
ok = [j for j in range(nb_ech) if admission[j]]

model.addConstrs((
    quicksum(c[j,i] for i in range(nb_notes)) + x[j] + epsilon == lmbda
    ) for j in rejected
)
model.addConstrs((
    quicksum(c[j,i] for i in range(nb_notes)) == lmbda + y[j]
    ) for j in ok
)

model.addConstrs((alpha <= x[j]) for j in range(nb_ech))
model.addConstrs((alpha <= y[j]) for j in range(nb_ech))
model.addConstrs((c[j,] <= w) for j in range(nb_ech))
model.addConstrs((c[j,] <= d[j,]) for j in range(nb_ech))
model.addConstrs((c[j,] >= d[j,] - np.ones(nb_notes) + w) for j in range(nb_ech))
model.addConstrs(((M*d[j,] + epsilon*np.ones(nb_notes) >= grades[j,] - b) for j in range(nb_ech)))
model.addConstrs(((M*(d[j,] - np.ones(nb_notes)) <= grades[j,] - b) for j in range(nb_ech)))

model.addConstr(quicksum(w[k] for k in range(nb_notes)) == 1)

model.update()
model.setObjective(alpha, GRB.MAXIMIZE)

# Résolution du PL
model.params.outputflag = 0 # (mode mute)
model.optimize()

if model.status != GRB.OPTIMAL:
    print("cannot converge")
    sys.exit()

print(f"""
Results:
- alpha: {alpha.X}
- lambda: {lmbda.X}
- w: {w.X}
- b: {b.X}
""")

ok = grades > b.X
res = (ok*w.X).sum(axis=1) > lmbda.X
print(dict(Counter(res)))

count = 0
for i in range(len(res)):
    if res[i] == admission[i]:
        count += 1
print(count/len(res))
