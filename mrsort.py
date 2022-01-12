from gurobipy import *
import numpy as np
from numpy.core.fromnumeric import shape
from itertools import product
from time import time

np.set_printoptions(precision=2)

class MRSort:
    def __init__(self, generator):
        self.gen = generator
        self.nb_split = generator.num_classes - 1
        self.grades, self.admission = generator.grades, generator.admission
        self.model = Model("MR-sort")
        self.objective = None

        # Constants
        self.nb_ech = self.gen.size
        self.nb_notes = self.gen.num_criterions

        # Gurobi variables
        self.alpha = self.model.addVar()
        self.x = self.model.addMVar(shape=self.nb_ech) # slack for each student (in A*)
        self.y = self.model.addMVar(shape=self.nb_ech) # slack for each student (in R*)
        self.w = self.model.addMVar(shape=self.nb_notes, lb=0, ub=1)
        self.b = self.model.addMVar(shape=(self.nb_notes, self.nb_split))

        self.lmbda = self.model.addVar(lb=0.5, ub=1)

        self.c = self.model.addMVar(shape=(self.nb_ech, self.nb_notes, self.nb_split), lb=0, ub=1)
        self.d = self.model.addMVar(shape=(self.nb_ech, self.nb_notes, self.nb_split), vtype=GRB.BINARY)

    def solve(self):
        start = time()
        if self.objective == None:
            return (None, 0)

        self.model.update()
        self.model.setObjective(self.objective, GRB.MAXIMIZE)
        self.model.params.outputflag = 0 # (mode mute)
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return (None, 0)
        compute_time = time() - start
        res = np.zeros((self.nb_ech))
        for i in range(self.nb_split):
            res += ((self.grades > self.b.X[:,i])*self.w.X).sum(axis=1) > self.lmbda.X
        return res, compute_time

    def print_params(self):
        print(f"Parametres trouves par MR-Sort:\n",
            f"- alpha: {self.alpha.X}\n",
            f"- lambda: {self.lmbda.X}\n",
            f"- weights: {self.w.X}\n",
        )


    def set_constraint(self):
        epsilon = 1e-9
        M = 1e2 # superieur a l'ecart max, 20

        self.model.addConstrs((
            quicksum(self.c[j,i,h] for i in range(self.nb_notes)) + self.x[j] + epsilon == self.lmbda
            ) for j,h in product(range(self.nb_ech), range(self.nb_split)) if h == self.admission[j]
        )
        self.model.addConstrs((
            quicksum(self.c[j,i,h] for i in range(self.nb_notes)) == self.lmbda + self.y[j]
            ) for j,h in product(range(self.nb_ech), range(self.nb_split)) if h+1 == self.admission[j]
        )
        self.model.addConstrs((self.alpha <= self.x[j]) for j in range(self.nb_ech))
        self.model.addConstrs((self.alpha <= self.y[j]) for j in range(self.nb_ech))
        self.model.addConstrs((
            self.c[j,:,h] <= self.w + epsilon)
            for j,h in product(range(self.nb_ech), range(self.nb_split))if h == self.admission[j] or h+1 == self.admission[j]
        )
        self.model.addConstrs((
            self.c[j,:,h] <= self.d[j,:,h])
            for j,h in product(range(self.nb_ech), range(self.nb_split)) if h == self.admission[j] or h+1 == self.admission[j]
        )
        self.model.addConstrs((
            self.c[j,:,h] >= self.d[j,:,h] - np.ones(self.nb_notes) + self.w)
            for j,h in product(range(self.nb_ech), range(self.nb_split)) if h == self.admission[j] or h+1 == self.admission[j]
        )
        self.model.addConstrs((
            M*self.d[j,:,h] + epsilon*np.ones(self.nb_notes) >= self.grades[j,] - self.b[:,h])
            for j,h in product(range(self.nb_ech), range(self.nb_split)) if h == self.admission[j] or h+1 == self.admission[j]
        )
        self.model.addConstrs((
            (M*(self.d[j,:,h] - np.ones(self.nb_notes)) <= self.grades[j,] - self.b[:,h]))
            for j,h in product(range(self.nb_ech), range(self.nb_split)) if h == self.admission[j] or h+1 == self.admission[j]
        )
        self.model.addConstr(quicksum(self.w[k] for k in range(self.nb_notes)) == 1)

        self.objective = self.alpha


