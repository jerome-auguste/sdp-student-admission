#%%
from gurobipy import *
import numpy as np
from numpy.core.fromnumeric import shape
from generator import Generator

#%%
gen = Generator(size=10, lmbda=12, weights=np.array([0.2, 0.4, 0.3, 0.1]), frontier=0.5)
ech = gen.generate()
print(ech)

exit
#%%
model = Model("MR-sort")

# Constants
nb_ech = 10
nb_notes = 4
epsilon = 0.01
eval_min = 12
lmbda = 12

# Gurobi variables
alpha = model.addVar() # min slack
x = model.addMVar(shape=nb_ech) # slack for each student (in A*)
y = model.addMVar(shape=nb_ech) # slack for each student (in R*)

c = model.addMVar(shape=(nb_ech, nb_notes))

# Matrix to express constraints
one_vector = np.ones((nb_notes, 1))
one_line = np.ones((1, nb_ech))

# Constraints
model.addConstr(c @ one_vector + x + epsilon*one_vector == lmbda*one_vector)

model.update()