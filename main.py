#%%
from gurobipy import *
from generator import Generator

grades,admission = Generator(size=1000).generate()
model = Model("MR-sort")
