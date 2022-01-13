"""This module solves an NCS problem with a SAT Solver (gophersat.exe)"""

import numpy as np

from generator import Generator
from utils import *

class NcsSatModel:
    """Non Compensatory Sorting model solved with (gophersat) SAT solver
    (cf. Belahcène et al 2018)"""

    def __init__(self, generator: Generator, train_set: np.ndarray,
                 labels: np.ndarray) -> None:

        # Generator attributes
        self.train_set = train_set
        self.labels = labels
        self.gen = generator

        # Reformatting variables
        self.coalitions = [tuple(el) for el in subsets(list(range(self.gen.num_criterions)))]
            # Tuple format is accepted as a key to the encoder dictionnary
        self.values_support = possible_values_per_crit(self.train_set)
            # Set of the possible values in the train_set for each criterion
        self.alternatives_per_class = [[u for u in range(self.gen.size) if self.labels[u] == h]
                            for h in range(self.gen.num_classes)]

        # Generating triplet (i, h, k) and coalition set B as mentioned in
        # Section 3.4,Definition 4 (SAT encoding for U-NCS)
        self.variables = {
            "frontier_var": list({(i, h, k) for i in range(self.gen.num_criterions)
                             for h in range(1, self.gen.num_classes)
                             for k in self.values_support[i]}),
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

    def clauses_2a(self) -> list:
        """Computes ascending scales clauses (named 2a in Definition 4)
        For all criteria i, classes h and adjacent pairs of value k<k':
        x_{i, h, k} => x_{i, h, k'}

        Returns:
            list: clauses according to the formula
        """
        clauses_2a = []

        # Not only adjacent values of k
        # for i in range(self.gen.num_criterions):
        #     crit_values = sorted(self.values_support[i])
        #     for h in range(self.gen.num_classes):
        #         for ik in range(len(crit_values)-1):
        #             for ikp in range(ik + 1, len(crit_values[ik + 1:])):
        #                 if crit_values[ik] < crit_values[ikp]:
        #                     clauses_2a.append([
        #                         self.front_v2i[(i, h, crit_values[ikp])],
        #                         -self.front_v2i[(i, h, crit_values[ik])]
        #                     ])
        #                     # print(
        #                     #     f"({i}, {h}, {crit_values[ikp]}) >
        #                     #          ({i}, {h}, {crit_values[ik]})"
        #                     # )

        # Only for adjacent values of k
        for i in range(self.gen.num_criterions):
            crit_values = self.values_support[i] # Values are unique and already sorted
            for h in range(1, self.gen.num_classes):
                for ik in range(len(crit_values) - 1):
                    ikp = ik + 1
                    clauses_2a.append([
                        self.front_v2i[(i, h, crit_values[ikp])],
                        -self.front_v2i[(i, h, crit_values[ik])],
                    ])

        return clauses_2a

    def clauses_2b(self) -> list:
        """Computes Hierarchy of profiles clauses (named 2b in Definition 4)
        (Evaluates classes (frontier) according to each value)
        For all criteria i, adjacent pairs of classes h<h', values k:
        x_{i, h', k} => x_{i, h, k} (being in higher classes makes the value greater to lower classes frontier)
        
        Returns:
            list: clauses according to the formula
        """
        # 3b Hierarchy of profiles
        clauses_2b = []

        # Not only for adjacent values
        # for i in range(self.gen.num_criterions):
        #     for k in set(self.values_support[i]):
        #         for h in range(self.gen.num_classes-1):
        #             for hp in range(h+1, self.gen.num_classes):
        #                 clauses_2b.append([self.front_v2i[(i, h, k)], -self.front_v2i[(i, hp, k)]])
                        # print(f"({i}, {h}, {k}) < ({i}, {hp}, {k})")

        # Only for adjacent values
        for i in range(self.gen.num_criterions):
            for h in range(1, self.gen.num_classes - 1):
                for k in set(self.values_support[i]):
                    clauses_2b.append(
                        [self.front_v2i[(i, h, k)], -self.front_v2i[(i, h + 1, k)]])
        return clauses_2b

    def clauses_2c(self) -> list:
        """Computes coalitions strength clauses (named 2c in Definition 4)
        For all "adjacent" (difference is exactly 1 element)
        pairs of coalitions (of criteria) B included in B'
        y_B => y_{B'} (as having more criteria still forma sufficient coalition)

        Returns:
            list: clauses according to the formula
        """
        clauses_2c = []

        # Not only for "adjacent" coalitions (difference is a singleton)
        # for B in self.coalitions:
        #     for Bp in self.coalitions:
        #         if set(B).issubset(set(Bp)):
        #             clauses_2c.append([self.coal_v2i[Bp], -self.coal_v2i[B]])
        #             # print(f"{B} is subset of {Bp}")

        # Only for a "adjacent" coalitions
        for B in self.coalitions:
            N_minus_B = {crit for crit in range(self.gen.num_criterions)} - set(B)
            for i in N_minus_B: # Adds exactly one element to the coalition
                Bp = set(B).union(set([i]))
                clauses_2c.append([self.coal_v2i[tuple(Bp)], -self.coal_v2i[B]])

        return clauses_2c

    def clauses_2d(self) -> list:
        """Computes alternatives outranked by boundary above them clauses (named 2d in Definition 4)
        (Ensures the correct representation of the assignment (labels))
        For all coalition B, for all frontier h and all datapoint u assigned to class h-1
        (AND_{i in B} x_{i, h, u_i}) => -y_B
        (if an alternative is predicted above the h frontier, then the coalition is not sufficient)

        Returns:
            list: clauses according to the formula
        """
        clauses_2d = []
        for B in self.coalitions:
            for h in range(1, self.gen.num_classes):
                for u in self.alternatives_per_class[h - 1]:
                    clauses_2d.append([-self.front_v2i[(i, h, self.train_set[u, i])]
                                    for i in B] + [-self.coal_v2i[B]])

        return clauses_2d

    def clauses_2e(self) -> list:
        """Computes alternatives outranked by boundary bellow them clauses
        (named 2e in Definition 4)
        (Ensures the correct representation of the assignment (labels))
        For all coalition B, for all frontier h and all datapoint a assessed at class h
        (AND_{i in B} -x_{i, h, a_i}) => y_{N-B}
        (If the alternative is assigned to the right class but all of its values are below the frontier in the coalition
        then the complementary coalition is sufficient)
        
        Returns:
            list: clauses according to the formula
        """
        clauses_2e = []
        N = set(list(range(self.gen.num_criterions)))
        for B in self.coalitions:
            for h in range(1, self.gen.num_classes):
                for a in self.alternatives_per_class[h]:
                    clauses_2e.append([self.front_v2i[(i, h, self.train_set[a, i])]
                                    for i in B] + [self.coal_v2i[tuple(N - set(B))]])
        return clauses_2e


    def run_solver(self) -> list:
        """Uses clasues defined above to encode the NCS problem
        into a SAT problem, solved by gophersat

        Returns:
            list: resulting frontiers between classes
        """
        my_clauses = self.clauses_2a() + self.clauses_2b() + self.clauses_2c(
        ) + self.clauses_2d() + self.clauses_2e()
        my_dimacs = clauses_to_dimacs(
            my_clauses,
            len(self.variables["frontier_var"]) + len(self.variables["coalition_var"]))

        write_dimacs_file(my_dimacs, "workingfile.cnf")
        res = exec_gophersat("workingfile.cnf")

        return res

    def solve(self):
        """Use gophersat to solve all the clauses defined in previous methods

        Returns:
            tuple: frontier and possible coalitions
        """
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

        # print("Frontier variables assumptions:")
        # for i in range(len(front_results)):
        #     print(front_results[i])

        # print(f"Resulted sufficient coalitions: {coal_results}")

        frontier = {}
        for h in range(1, self.gen.num_classes):
            class_front = []
            for i in range(self.gen.num_criterions):
                criterion_val = [x[2] for x in front_results if x[0] == i and x[1] == h]
                if len(criterion_val) == 0: # In case no frontier is found for this criteria, remove it from coalitions
                    coal_results = [coal for coal in coal_results if i not in coal]
                    crit_front = None
                else:
                    crit_front = min(criterion_val) # (i, h, k)
                # if len(crit_val)>0:
                #     crit_front = min([x[2] for x in front_results if x[0] == i and x[1] == h]) # (i, h, k)
                # else:
                #     print("NO VALUE FOUND FOR THIS CRIT AND THIS CLASS")
                # class_front.append(
                #     min(crit_res) if len(crit_res) > 0 else
                #     (max([self.train_set[u][i] for u in self.alternatives_per_class[h]]) -
                #      min([self.train_set[u][i] for u in self.alternatives_per_class[h]])) * h /
                #     self.gen.num_classes + min(self.alternatives_per_class[h]))
                
                class_front.append(crit_front)
                
                # To fix error when no boundary is found for a specific class
            frontier[h] = class_front
        # print("\nFrontier")
        # for el in frontier:
        #     print(el)

        # for coal in coal_results:
        #     print(f"For coalition: {coal}")
        #     coal_clf = []
        #     for i_stud, student in enumerate(self.train_set):
        #         clf = 0
        #         stud_clf = []
        #         for crit in range(self.gen.num_criterions):
        #             if crit in coal:
        #                 # print(f"student[crit] = {student[crit]} \t frontier[h_1][crit] = {frontier[:][crit]} \t num_classes = {self.gen.num_classes}")
        #                 clf += sum([student[crit] >= frontier[h_1][crit] for h_1 in range(self.gen.num_classes-1)])
        #                 stud_clf.append(clf)
        #         pred = 0
        #         for h in range(self.gen.num_classes):
        #             pred += sum(stud_clf[i] >= h)
        #         # pred = round(clf/self.gen.num_criterions)
        #         pred = 0
        #         coal_clf.append(pred == int(self.labels[i_stud]))
        #         # print(f"Predicted class: {pred} \t Real class: {int(self.labels[i_stud])}")
        #     print(f"Accuracy = {sum(coal_clf)/len(coal_clf)*100:.0f} %")

        # print(f"Resulted frontiers: {frontier}")
        best_pred = [0]*self.gen.size
        best_accuracy = 0
        for coal in coal_results:
            coal_pred = []
            for alt in self.train_set:
                alt_pred = []
                for i in coal:
                    alt_pred.append(sum([alt[i] >= frontier[h][i] for h in range(1, self.gen.num_classes)])) # Vote for each criterion of the coalition
                coal_pred.append(min(alt_pred)) # Classes are ordered so the min is the one valid for the entire coalition
                
            coal_accuracy = sum([coal_pred[s] == self.labels[s] for s in range(self.gen.size)])/self.gen.size
            if coal_accuracy > best_accuracy:
                best_accuracy = coal_accuracy
                best_pred = coal_pred
        
        
        return best_pred

# Quelles sont les valeurs possibles pour les notes k ? Entier uniquement ?
