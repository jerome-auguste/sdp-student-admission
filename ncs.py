"""This module solves an NCS problem with a SAT Solver (gophersat.exe)"""

import numpy as np

from generator import Generator
from utils import *

class NcsSatModel:
    """Non Compensatory Sorting model solved with gophersat SAT solver
    (cf. https://arxiv.org/pdf/1710.10098.pdf)"""

    def __init__(self, generator: Generator, train_set: np.ndarray,
                 labels: np.ndarray) -> None:

        # Generator attributes
        self.train_set = train_set
        self.labels = labels
        self.gen = generator

        # Reformatting variables
        self.coalitions = [tuple(el) for el in subsets(list(range(self.gen.num_criterions)))]
        self.grades_support = grades_support_per_crit(self.train_set)
        self.datapoints_per_class = [[u for u in range(self.gen.size) if self.labels[u] == h]
                            for h in range(self.gen.num_classes)]

        # Generating triplet (i, h, k) values as mentioned in
        # Section 3.4,Definition 4 (SAT encoding for U-NCS)
        self.variables = {
            "frontier_var": [(i, h, k) for i in range(self.gen.num_criterions)
                             for h in range(self.gen.num_classes)
                             for k in self.grades_support[i]],
            "coalition_var":
            self.coalitions,
        }

        # Encodes/Decodes variables defining the frontiers
        # to/from int index (necessary according to gophersat)
        self.front_v2i = {v: i + 1 for i, v in enumerate(self.variables["frontier_var"])}
        self.front_i2v = {i: v for v, i in self.front_v2i.items()}

        # Encodes/Decodes variables defining the sufficient coalitions to/from int index
        self.coal_v2i = {
            v: i + len(self.front_i2v) + 1
            for i, v in enumerate(self.variables["coalition_var"])
        }  # Indexes are starting right above where frontier indexing stops
        self.coal_i2v = {i: v for v, i in self.coal_v2i.items()}

        # Concatenates both encoding for the solver
        self.i2v = {}
        self.i2v.update(self.front_i2v)
        self.i2v.update(self.coal_i2v)

    def clauses_3a(self) -> list:
        """Computes ascending scales clauses (named 3a in Definition 4)
        (Evaluates values according to the frontiers (for each criterion))
        For all criteria i, classes h and adjacent pairs of value k<k':
        x_{i, h, k'} or -x_{i, h, k}

        Returns:
            list: clauses according to the formula
        """
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
        #                         f"({i}, {h}, {sorted_grades[ikp]}) >
        #                              ({i}, {h}, {sorted_grades[ik]})"
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
        """Computes Hierarchy of profiles clauses (names 3b in Definition 4)
        (Evaluates classes (frontier) according to each value)
        For all criteria i, adjacent pairs of classes h<h', values k:
        x_{i, h, k} or -x_{i, h', k}
        
        Returns:
            list: clauses according to the formula
        """
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
        """Computes coalitions strength clauses (named 3c in Definition 4)
        (Evaluates valid coalitions of criteria to classify the datapoints)
        For all "adjacent" (difference is exactly 1 element)
        pairs of coalitions (of criteria) B included in B'
        y_{B'} or -y_B

        Returns:
            list: clauses according to the formula
        """
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
        """Computes alternatives outranked by boundary above them clauses (named 3d in Definition 4)
        (Ensures the correct representation of the assignment (labels))
        For all coalition B, for all frontier h and all datapoint u assessed at class h-1
        (OR_{i in B} -x_{i, h, u_i}) or -y_B

        Returns:
            list: clauses according to the formula
        """
        clauses_3d = []
        for B in self.coalitions:
            for h in range(1, self.gen.num_classes):
                for u in self.datapoints_per_class[h - 1]:
                    clauses_3d.append([-self.front_v2i[(i, h, self.train_set[u, i])]
                                    for i in B] + [-self.coal_v2i[B]])

        return clauses_3d

    def clauses_3e(self) -> list:
        """Computes alternatives outranked by boundary bellow them clauses
        (named 3d in Definition 4)
        (Ensures the correct representation of the assignment (labels))
        For all coalition B, for all frontier h and all datapoint a assessed at class h-1
        (OR_{i in B} x_{i, h, a_i}) or y_{N-B}

        Returns:
            list: clauses according to the formula
        """
        clauses_3e = []
        for B in self.coalitions:
            for h in range(self.gen.num_classes):
                for a in self.datapoints_per_class[h]:
                    N_minus_B = tuple(set(list(range(self.gen.num_criterions))) - set(B))
                    clauses_3e.append([self.front_v2i[(i, h, self.train_set[a, i])]
                                    for i in B] + [self.coal_v2i[N_minus_B]])
        return clauses_3e


    def run_solver(self) -> list:
        """Uses clasues defined above to encode the NCS problem
        into a SAT problem, solved by gophersat

        Returns:
            list: resulting frontiers between classes
        """
        my_clauses = self.clauses_3a() + self.clauses_3b() + self.clauses_3c(
        ) + self.clauses_3d() + self.clauses_3e()
        my_dimacs = clauses_to_dimacs(
            my_clauses,
            len(self.variables["frontier_var"]) + len(self.variables["coalition_var"]))

        write_dimacs_file(my_dimacs, "workingfile.cnf")
        res = exec_gophersat("workingfile.cnf")
        
        return res
    
    def solve(self):
        res = self.run_solver()
        
        # Results
        is_sat, model = res
        index_model = [int(x) for x in model if int(x) != 0]
        var_model = {
            self.i2v[abs(int(v))]: int(v) > 0
            for v in model if int(v) != 0
        }
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
        print("\nFrontier")
        for el in frontier:
            print(el)
        
        for coal in coal_results:
            print(f"For coalition: {coal}")
            coal_clf = []
            for i_stud, student in enumerate(self.train_set):
                clf = 0
                for crit in range(self.gen.num_criterions):
                    if crit in coal:
                        # print(f"student[crit] = {student[crit]} \t frontier[h_1][crit] = {frontier[:][crit]} \t num_classes = {self.gen.num_classes}")
                        clf += sum([student[crit] > frontier[h_1][crit] for h_1 in range(self.gen.num_classes-1)])
                pred = round(clf/self.gen.num_criterions)
                coal_clf.append(pred == int(self.labels[i_stud]))
                # print(f"Predicted class: {pred} \t Real class: {int(self.labels[i_stud])}")
            print(f"Accuracy = {sum(coal_clf)/len(coal_clf)*100:.0f} %")
                

        # print(f"Resulted frontiers: {frontier}")
        return frontier, coal_results

# Quelles sont les valeurs possibles pour les notes k ? Entier uniquement ?

# TODO: score results
