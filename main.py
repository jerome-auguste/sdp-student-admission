#%%
from gurobipy import *
import numpy as np
from numpy.core.fromnumeric import shape
from generator import Generator

#%%
from generator import Generator
grades,admission = Generator(size=1000).generate()

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