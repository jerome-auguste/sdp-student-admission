"""This module solves an U-NCS problem with a SAT Solver (gophersat)"""

from tools.generator import Generator
from tools.utils import sorted_alternatives_per_crit, subsets, clauses_to_dimacs, write_dimacs_file, exec_gophersat


class MaxSatSinglePeakModel:
    """Non Compensatory Sorting model solved with (gophersat) MaxSAT solver
    (cf. Tlili et al 2022)"""

    def __init__(self, generator: Generator) -> None:

        # Generator attributes
        self.gen = generator
        self.train_set = self.gen.grades
        self.labels = self.gen.admission

        # Reformatting variables
        self.coalitions = [
            tuple(el) for el in subsets(list(range(self.gen.num_criteria)))
        ]
        # Tuple format is accepted as a key to the encoder dictionnary
        self.sorted_alt = sorted_alternatives_per_crit(self.train_set)
        # Set of the possible values in the train_set for each criterion
        self.alternatives_per_class = [[
            u for u in range(len(self.train_set)) if self.labels[u] == h
        ] for h in range(self.gen.num_classes)]

        # Generating triplet (i, h, k) and coalition set B as mentioned in
        # Section 3.4,Definition 4 (SAT encoding for U-NCS)
        self.variables = {
            "frontier_var":
            list((i, h, x)
                  for i in range(self.gen.num_criteria)
                  for h in range(1, self.gen.num_classes)
                  for x in self.sorted_alt[i]),
            "coalition_var":
            self.coalitions,
        }

        # Encodes/Decodes variables defining the frontiers
        # to/from int index (necessary according to gophersat)
        self.front_v2i = {
            v: i + 1
            for i, v in enumerate(self.variables["frontier_var"])
        }
        self.front_i2v = {i: v for v, i in self.front_v2i.items()}

        # Encodes/Decodes variables defining the sufficient coalitions to/from int index
        self.coal_v2i = {
            v: i + len(self.front_i2v) + 1
            for i, v in enumerate(self.variables["coalition_var"])
        }  # Indexes are starting right above where frontier indexing stops
        self.coal_i2v = {i: v for v, i in self.coal_v2i.items()}
        
        # Encodes/Decodes triggered if the NCS correctly classifies an alternative
        self.correct_clf_v2i = {
            z: z + len(self.front_i2v) + len(self.coal_i2v) + 1
            for z in range(len(self.train_set))}
        
        self.correct_clf_i2v = {i: v for v, i in self.coal_v2i.items()}

        # Concatenates both encoding for the solver
        self.i2v = {}
        self.i2v.update(self.front_i2v)
        self.i2v.update(self.coal_i2v)
        self.i2v.update(self.correct_clf_i2v)

        # Results to be shared to predict
        self.frontier = {i: [0]*self.gen.num_criteria for i in range(1, self.gen.num_classes)}
        self.suff_coal = ()

        self.gopherpath = None

    def set_gophersat_path(self, gopherpath):
        self.gopherpath = gopherpath

    def clauses_c1(self) -> list:
        """Computes ascending scales clauses (named C1),
        those clauses are considered hard from weights point of view
        For all criteria i, classes k and values x<x'<x":
        x_{i, k, x} and x_{i, k, x"} => x_{i, k, x'}

        Returns:
            list: clauses according to the formula
        """
        clauses_c1 = []
        # Only for adjacent values of k
        for i in range(self.gen.num_criteria):
            crit_values = self.sorted_alt[
                i]  # Values are unique and already sorted
            for h in range(1, self.gen.num_classes):
                for ix in range(len(crit_values) - 2):
                    ixp = ix + 1
                    ixs = ix + 2
                    clauses_c1.append([
                        self.front_v2i[(i, h, crit_values[ixp])],
                        -self.front_v2i[(i, h, crit_values[ix])],
                        -self.front_v2i[(i, h, crit_values[ixs])]
                    ])

        return clauses_c1

    def clauses_c2(self) -> list:
        """Computes Hierarchy of profiles clauses (named 2b in Definition 4)
        (Evaluates classes (frontier) according to each value),
        those clauses are considered hard from weights point of view
        For all criteria i, adjacent pairs of classes k<k', alternatives x:
        a_{i, k', x} => a_{i, k, x}
        (being in higher classes makes the value greater to lower classes frontier)

        Returns:
            list: clauses according to the formula
        """
        # 3b Hierarchy of profiles
        clauses_c2 = []
        # Only for adjacent values
        for i in range(self.gen.num_criteria):
            for h in range(1, self.gen.num_classes - 1):
                for x in set(self.sorted_alt[i]):
                    clauses_c2.append([
                        self.front_v2i[(i, h, x)],
                        -self.front_v2i[(i, h + 1, x)]
                    ])
        return clauses_c2

    def clauses_c3(self) -> list:
        """Computes coalitions strength clauses (named 2c in Definition 4),
        those clauses are considered hard from weights point of view
        For all "adjacent" (difference is exactly 1 element)
        pairs of coalitions (of criteria) B included in B'
        t_B => t_{B'} (as having more criteria still forms a sufficient coalition)

        Returns:
            list: clauses according to the formula
        """
        clauses_c3 = []
        # Only for a "adjacent" coalitions
        for B in self.coalitions:
            N_minus_B = {crit
                         for crit in range(self.gen.num_criteria)} - set(B)
            for i in N_minus_B:  # Adds exactly one element to the coalition
                Bp = set(B).union(set([i]))
                clauses_c3.append(
                    [self.coal_v2i[tuple(Bp)], -self.coal_v2i[B]])

        return clauses_c3

    def clauses_c5(self) -> list:
        """Computes alternatives outranked by boundary above them clauses (named c5 tilde)
        (Ensures the correct representation of the assignment (labels)),
        those clauses are considered soft from weights point of view
        For all coalition B, for all frontier k and all datapoint x assigned to class k-1
        (AND_{i in B} a_{i, k, x}) => -y_B OR -z_x
        (if an alternative is predicted above the k frontier, then the coalition is not sufficient
        or the alternative is part of the noise)

        Returns:
            list: clauses according to the formula
        """
        clauses_c5 = []
        for B in self.coalitions:
            for k in range(1, self.gen.num_classes):
                for x in self.alternatives_per_class[k - 1]:
                    clauses_c5.append([
                        -self.front_v2i[(i, k, x)]
                        for i in B
                    ] + [-self.coal_v2i[B]] + [-self.correct_clf_v2i[x]])

        return clauses_c5

    def clauses_c6(self) -> list:
        """Computes alternatives outranked by boundary bellow them clauses
        (named 2e in Definition 4)
        (Ensures the correct representation of the assignment (labels)),
        those clauses are considered soft from weights point of view
        For all coalition B, for all frontier h and all datapoint x assessed at class h
        (AND_{i in B} -a_{i, h, x}) => y_{N-B} OR -z_x
        (If the alternative is assigned to the right class
        whereas all of its values are below the frontier for the considered coalition
        then the complementary coalition is sufficient or the datapoint is part of the noise)

        Returns:
            list: clauses according to the formula
        """
        clauses_c6 = []
        N = set(list(range(self.gen.num_criteria)))
        for B in self.coalitions:
            for h in range(1, self.gen.num_classes):
                for x in self.alternatives_per_class[h]:
                    clauses_c6.append([
                        self.front_v2i[(i, h, x)] for i in B
                    ] + [self.coal_v2i[tuple(N - set(B))]] + [-self.correct_clf_v2i[x]])
        return clauses_c6
    
    def clauses_goal(self) -> list:
        """Goal of the MaxSat: correctly classify most of the alternatives

        Returns:
            list: all triggers for all alternatives
        """
        return [self.correct_clf_v2i[x] for x in range(len(self.train_set))]


    def run_solver(self) -> list:
        """Uses clasues defined above to encode the NCS problem
        into a SAT problem, solved by gophersat

        Returns:
            list: resulting frontiers between classes
        """
        # Add weights to the clauses
        hard_clauses = self.clauses_c1() + self.clauses_c2() + self.clauses_c3() + self.clauses_c5() + self.clauses_c6()
        soft_clauses = self.clauses_goal()
        hard_weight = len(soft_clauses) + 1
        
        for i, clause in enumerate(hard_clauses):
            hard_clauses[i] = [hard_weight] + clause
        
        for i, clause in enumerate(soft_clauses):
            soft_clauses[i] = [1] + [clause]
        
        my_clauses = hard_clauses + soft_clauses
        
        my_dimacs = clauses_to_dimacs(
            my_clauses,
            len(self.variables["frontier_var"]) +
            len(self.variables["coalition_var"]), max_weight=hard_weight)

        write_dimacs_file(my_dimacs, "workingfile.wcnf")
        res = exec_gophersat(filename="workingfile.wcnf", cmd=self.gopherpath, weighted=True)

        return res

    def train(self):
        """Trains model to find the best coalition and frontier that matches the train_set

        Returns:
            tuple: frontier and possible coalitions
        """
        res = self.run_solver()

        # Results
        is_sat, model = res
        if not is_sat:
            print("--------------------------------------- SAT WARNING! ---------------------------------------")
            print("-              Optimum not found, alternatives might be assigned to class 0                -")
            print("--------------------------------------------------------------------------------------------")



        # index_model = [int(x) for x in model if int(x) != 0]
        var_model = {
            self.i2v[abs(int(v))]: int(v) > 0
            for v in model if int(v) != 0
        }
        front_results = [
            x for x in self.variables["frontier_var"] if x in var_model and var_model[x]
        ]
        coal_results = [
            x for x in self.variables["coalition_var"] if x in var_model and var_model[x]
        ]

        # print("Frontier variables assumptions:")
        # for i in range(len(front_results)):
        #     print(front_results[i])

        # print(f"Resulted sufficient coalitions: {coal_results}")

        # frontier = {i: [0]*self.gen.num_criteria for i in range(1, self.gen.num_classes)}
        for h in range(1, self.gen.num_classes):
            class_front = [(0, 0)]*self.gen.num_criteria
            for i in range(self.gen.num_criteria):
                criterion_val = [
                    x[2] for x in front_results if x[0] == i and x[1] == h
                ]
                if len(criterion_val) == 0:
                    # In case no frontier is found for this criterion, remove it from coalitions
                    coal_results = [
                        coal for coal in coal_results if i not in coal
                    ]
                    
                else:
                    min_crit = min(criterion_val)  # (i, h, k)
                    max_crit = max(criterion_val)
                    class_front[i] = (min_crit, max_crit)

            self.frontier[h] = class_front
        # print("\nFrontier")
        # for el in frontier:
        #     print(el)

        # Find the best coalition for the considered frontier
        best_pred = [0] * self.gen.size
        best_accuracy = 0
        best_coal = tuple()
        for coal in coal_results:
            coal_pred = []
            for alt in self.train_set:
                alt_pred = []
                for i in coal:
                    alt_pred.append(
                        sum([
                            (alt[i] >= self.frontier[h][i][0]) and (alt[i] <= self.frontier[h][i][1])
                            for h in range(1, self.gen.num_classes)
                        ]))  # Vote for each criterion of the coalition
                coal_pred.append(
                    min(alt_pred)
                )  # Classes are ordered so the min is the one valid for the entire coalition

            coal_accuracy = sum(
                [coal_pred[s] == self.labels[s]
                 for s in range(len(self.train_set))]) / len(self.train_set)
            if coal_accuracy > best_accuracy:
                best_accuracy = coal_accuracy
                best_pred = coal_pred
                best_coal = coal
                self.suff_coal = best_coal
        return best_pred

    def predict(self):
        """Predicts labels (classes) from a test_set of students
        The test_set has to have the same .shape[1] than the train_set used to train the model

        Returns:
            list: labels (classes) of the train_set (len(pred) == test_set.shape[0])
        """
        pred = [0]*len(self.gen.grades_test)
        for i_alt, alt in enumerate(self.gen.grades_test):
            try :
                pred[i_alt] = min([  # Takes the min class found (assuming ordered classes)
                        sum([  # Classifies values for each criteria
                            (alt[i] >= self.frontier[h][i][0]) and (alt[i] <= self.frontier[h][i][1])
                            for h in range(1, self.gen.num_classes) if h in self.frontier
                        ]) for i in self.suff_coal
                    ])
            except ValueError:
                pass


        return pred
