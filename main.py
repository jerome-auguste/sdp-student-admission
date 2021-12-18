#%%
from gurobipy import *
from generator import Generator

grades,admission = Generator(size=1000).generate()
model = Model("MR-sort")

M = 1000
EPSILON = 0.0001


# %%
