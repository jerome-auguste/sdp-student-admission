from gurobipy import *
import numpy as np
from numpy.core.fromnumeric import shape
from collections import Counter

np.set_printoptions(precision=2)

class MRSort:
    def __init__(self, generator):
        self.gen = generator
        self.num_classes = generator.num_classes
        self.grades, self.admission = generator.generate()
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
        self.b = self.model.addMVar(shape=self.nb_notes)

        self.lmbda = self.model.addVar(lb=0.5, ub=1)

        self.c = self.model.addMVar(shape=(self.nb_ech, self.nb_notes), lb=0, ub=1)
        self.d = self.model.addMVar(shape=(self.nb_ech, self.nb_notes), vtype=GRB.BINARY)
    
    def solve(self):
        if self.objective == None:
            return False
        self.model.update()
        self.model.setObjective(self.objective, GRB.MAXIMIZE)
        self.model.params.outputflag = 0 # (mode mute)
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return False
        return True

    def print_data(self):
        print(f"Parameters:\n",
            f"- lambda: {self.gen.lmbda}\n",
            f"- weights: {self.gen.weights}\n",
            f"- frontier: {self.gen.frontier}\n",
            f"- echantillons: {dict(Counter(self.admission))}\n"
        )

    def print_res(self):
        passing_grades = self.grades > self.b.X
        res = (passing_grades*self.w.X).sum(axis=1) > self.lmbda.X
  
        print(f"Resultats:\n",
            f"- alpha: {self.alpha.X}\n",
            f"- lambda: {self.lmbda.X}\n",
            f"- weights: {self.w.X}\n",
            f"- frontier: {self.b.X}\n",
            f"- resultats: {dict(Counter(res))}\n",
            f"- precision: {sum([res[i]==self.admission[i] for i in range(len(res))])/len(res)}\n"
        )
            

    def set_constraint(self):
        if self.num_classes == 2:
            epsilon = 1e-9
            M = 1e2 # superieur a l'ecart max, 20

            self.model.addConstrs((
                quicksum(self.c[j,i] for i in range(self.nb_notes)) + self.x[j] + epsilon == self.lmbda
                ) for j in range(self.nb_ech) if not self.admission[j]
            )
            self.model.addConstrs((
                quicksum(self.c[j,i] for i in range(self.nb_notes)) == self.lmbda + self.y[j]
                ) for j in range(self.nb_ech) if self.admission[j]
            )

            self.model.addConstrs((self.alpha <= self.x[j]) for j in range(self.nb_ech))
            self.model.addConstrs((self.alpha <= self.y[j]) for j in range(self.nb_ech))
            self.model.addConstrs((self.c[j,] <= self.w + epsilon) for j in range(self.nb_ech))
            self.model.addConstrs((self.c[j,] <= self.d[j,]) for j in range(self.nb_ech))
            self.model.addConstrs((self.c[j,] >= self.d[j,] - np.ones(self.nb_notes) + self.w) for j in range(self.nb_ech))
            self.model.addConstrs(((M*self.d[j,] + epsilon*np.ones(self.nb_notes) >= self.grades[j,] - self.b) for j in range(self.nb_ech)))
            self.model.addConstrs(((M*(self.d[j,] - np.ones(self.nb_notes)) <= self.grades[j,] - self.b) for j in range(self.nb_ech)))

            self.model.addConstr(quicksum(self.w[k] for k in range(self.nb_notes)) == 1)
        
            self.objective = self.alpha
        else:
            raise AttributeError()


