"""This module solves an NCS problem with a SAT Solver (gophersat.exe)"""

import subprocess

import numpy as np

from generator import Generator
from utils import grades_support_per_crit, subsets, clauses_to_dimacs, write_dimacs_file, exec_gophersat

# TRAIN_SET_SIZE = 50
# NUM_CLASSES = 3
# gen = Generator(size=TRAIN_SET_SIZE, num_classes=NUM_CLASSES)

# grades, admission = gen.generate()
# print(
#     f"Parameters:\nfrontier: {gen.frontier}\nelements: {dict(Counter(admission))}\n"
# )

class NcsSatModel:
    def __init__(self, generator: Generator, train_set: np.ndarray, labels: np.ndarray) -> None:
        
        # Generator attributes
        self.train_set = train_set
        self.labels = labels
        self.gen = generator
        
        # Reformatting variables
        self.coalitions = [tuple(el) for el in subsets(list(range(self.gen.num_criterions)))]
        self.grades_support = grades_support_per_crit(self.train_set)
        
        # Generating triplet values as mentioned in https://arxiv.org/pdf/1710.10098.pdf (Section 3.4, Definition 4 (SAT encoding for U-NCS))
        self.variables = {
            "frontier_var": [(i, h, k) for i in range(self.gen.num_criterions)
                             for h in range(self.gen.num_classes)
                             for k in self.grades_support[i]],
            "coalition_var":
            self.coalitions,
        }
        
        # Encodes/Decodes variables defining the frontiers
        self.front_v2i = {v: i + 1 for i, v in enumerate(self.variables["frontier_var"])}
        self.front_i2v = {i: v for v, i in self.front_v2i.items()}
        
        # Encodes/Decodes variables defining the sufficient coalitions
        self.coal_v2i = {
            v: i + len(self.front_i2v) + 1
            for i, v in enumerate(variables["coalition_var"])
        }  # Indexes are starting right above where frontier indexing stops
        self.coal_i2v = {i: v for v, i in self.coal_v2i.items()}

        # Concatenates both encoding
        self.i2v.update(self.front_i2v)
        self.i2v.update(self.coal_i2v)
    
    def clauses_3a(self) -> list:
        # 3a Ascending scales
        clauses_3a = []

        # Not only adjacent values of k
        # for i in range(gen.num_criterions):
        #     sorted_grades = sorted(grades_support[i])
        #     for h in range(gen.num_classes):
        #         for ik in range(len(sorted_grades)):
        #             for ikp in range(ik + 1, len(sorted_grades[ik + 1:])):
        #                 if sorted_grades[ik] < sorted_grades[ikp]:
        #                     clauses_3a.append([
        #                         front_v2i[(i, h, sorted_grades[ikp])],
        #                         -front_v2i[(i, h, sorted_grades[ik])]
        #                     ])
        #                     print(
        #                         f"({i}, {h}, {sorted_grades[ikp]}) >  ({i}, {h}, {sorted_grades[ik]})"
        #                     )

        # Only for adjacent values of k
        for i in range(self.gen.num_criterions):
            sorted_grades = sorted(self.grades_support[i])
            for h in range(self.gen.num_classes):
                for ik in range(len(sorted_grades) - 1):
                    ikp = ik + 1
                    while ikp < len(
                            sorted_grades) and sorted_grades[ik] >= sorted_grades[ikp]:
                        ikp += 1
                    if ikp < len(sorted_grades):
                        clauses_3a.append([
                            self.front_v2i[(i, h, sorted_grades[ikp])],
                            -self.front_v2i[(i, h, sorted_grades[ik])],
                        ])
        
        return clauses_3a

    def clauses_3b(self) -> list:
        # 3b Hierarchy of profiles
        clauses_3b = []

        # Not only for adjacent values
        # for i in range(gen.num_criterions):
        #     for k in set(grades_support[i]):
        #         for h in range(gen.num_classes-1):
        #             for hp in range(h+1, gen.num_classes):
        #                 clauses_3b.append([front_v2i[(i, h, k)], -front_v2i[(i, hp, k)]])
        #                 # print(f"({i}, {h}, {k}) < ({i}, {hp}, {k})")

        # Only for adjacent values
        for i in range(self.gen.num_criterions):
            for k in set(self.grades_support[i]):
                for h in range(self.gen.num_classes - 1):
                    clauses_3b.append(
                        [self.front_v2i[(i, h, k)], -self.front_v2i[(i, h + 1, k)]])
        
        return clauses_3b

    def clauses_3c(self) -> list:
        # 3c Coalitions strenghs

        clause_3c = []

        # Not only for "adjacent" coalitions (difference is a singleton)
        # for B in coalitions:
        #     for Bp in coalitions:
        #         if set(B).issubset(set(Bp)):
        #             clause_3c.append([coal_v2i[Bp], -coal_v2i[B]])
        #             print(f"{B} is subset of {Bp}")

        # Only for a "adjacent" coalitions
        for B in self.coalitions:
            for i in range(self.gen.num_criterions):
                Bp = set(B).union(set([i]))
                if Bp != set(B):
                    clause_3c.append([self.coal_v2i[tuple(Bp)], -self.coal_v2i[B]])
        
        return clause_3c

    def clauses_3d(self) -> list:
        # 3d Alternatives are outranked by boundary above them
        students_per_class = [[u for u in range(self.gen.size) if self.admission[u] == h]
                            for h in range(self.gen.num_classes)]  # Students per category

        clauses_3d = []
        for B in self.coalitions:
            for h in range(1, self.gen.num_classes):
                for u in students_per_class[h - 1]:
                    clauses_3d.append([-self.front_v2i[(i, h, self.grades[u, i])]
                                    for i in B] + [-self.coal_v2i[B]])
        
        return clauses_3d
    
    def clauses_3e(self) -> list:
        # 3e Alternatives outrank the boundary bellow them
        clauses_3e = []
        for B in coalitions:
            for h in range(gen.num_classes):
                for u in students_per_class[h]:
                    N_minus_B = tuple(set(list(range(gen.num_criterions))) - set(B))
                    clauses_3e.append([front_v2i[(i, h, grades[u, i])]
                                    for i in B] + [coal_v2i[N_minus_B]])
        return clauses_3e


    def solve(self) -> list:
        MY_CLAUSES = self.clauses_3a() + self.clauses_3b() + self.clause_3c() + self.clauses_3d() + self.clauses_3e()
        MY_DIMACS = clauses_to_dimacs(
            MY_CLAUSES,
            len(variables["frontier_var"]) + len(variables["coalition_var"]))

        write_dimacs_file(MY_DIMACS, "workingfile.cnf")
        res = exec_gophersat("workingfile.cnf")

        # Résultat
        is_sat, i_model, var_model = res
        front_results = [x for x in self.variables["frontier_var"] if var_model[x]]
        coal_results = [x for x in self.variables["coalition_var"] if var_model[x]]

        # print(f"Resulted sufficient coalitions: {coal_results}")

        frontier = []
        for h in range(1, self.gen.num_classes):
            class_front = []
            for i in range(self.gen.num_criterions):
                crit_res = [x[2] for x in front_results if x[0] == i and x[1] == h]
                if len(crit_res) > 0:
                    class_front.append(min(crit_res))
            frontier.append(class_front)

        # print(f"Resulted frontiers: {frontier}")
        return frontier

# Quelles sont les valeurs possibles pour les notes k ? Entier uniquement ?

# TODO: score results


