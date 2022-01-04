#%%
from gurobipy import *
import numpy as np
from numpy.core.fromnumeric import shape
from generator import Generator
from itertools import product

#%%
gen = Generator(size=1000)
grades,admission = gen.generate()

model = Model("MR-sort")

# Constants
nb_ech = gen.size
nb_notes = gen.num_criterions
epsilon = 0.01
eval_min = 12
M = 100

# A_mask = np.zeros((nb_ech, nb_ech))
# R_mask = np.zeros((nb_ech, nb_ech))
# for i in range(nb_ech):
#     if admission[i]:
#         A_mask[i,i] = True
#     else:
#         R_mask[i,i] = True

# Gurobi variables
alpha = model.addMVar(shape=1) # min slack
x = model.addMVar(shape=nb_ech) # slack for each student (in A*)
y = model.addMVar(shape=nb_ech) # slack for each student (in R*)
w = model.addMVar(shape=nb_notes, lb=0, ub=1)
b = model.addMVar(shape=nb_notes)

lmbda = model.addVar(lb=0.5, ub=1)

c = model.addMVar(shape=(nb_ech, nb_notes), lb=0, ub=1)
d = model.addMVar(shape=(nb_ech, nb_notes), vtype=GRB.BINARY)

# Matrix to express constraints
one_vector = np.ones((nb_ech, 1))
one_line = np.ones((1, nb_notes))

# Constraints
# Notes :
# - les contraintes doivent etre des 1D MVars objects
# - la multipplication element par element (a*b) semble
#   poser probleme avec les variables Gurobi

model.addConstrs((
    quicksum(c[j,i] for i in range(nb_notes)) + x[j] + epsilon == lmbda
    ) for j in range(nb_ech) if admission[j]
)
model.addConstrs((
    quicksum(c[j,i] for i in range(nb_notes)) == lmbda + y[j]
    ) for j in range(nb_ech) if not admission[j]
)

# Try with only matrix
# model.addConstr((c @ one_vector + x + epsilon*one_vector) @ A_mask == (lmbda*one_vector)*A_mask)
# model.addConstr((c @ one_vector) @ R_mask == (lmbda*one_vector + x) @ R_mask)

model.addConstr(one_vector @ alpha <= x)
model.addConstr(one_vector @ alpha <= y)
model.addConstrs((c[j,] <= w) for j in range(nb_ech))
model.addConstrs((c[j,] <= d[j,]) for j in range(nb_ech))
model.addConstrs((c[j,] >= d[j,] - nb_notes + w) for j in range(nb_ech))
model.addConstrs(((M*d[j,i] + epsilon*nb_notes >= w[i]*grades[j,i] - b[i]) for j,i in product(range(nb_ech), range(nb_notes))))
model.addConstrs(((M*(d[j,i] - 1) <= w[i]*grades[j,i] - b[i]) for j,i in product(range(nb_ech), range(nb_notes))))
model.addConstr(one_line @ w == 1)

model.update()
