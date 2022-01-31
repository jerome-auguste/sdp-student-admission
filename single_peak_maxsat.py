"""This module solves an U-NCS problem with a SAT Solver (gophersat)"""

from tools.generator import Generator
from tools.utils import possible_values_per_crit, subsets, clauses_to_dimacs, write_dimacs_file, exec_gophersat


class SinglePeakModel:
    """Non Compensatory Sorting model solved with (gophersat) SAT solver
    (cf. Belahcène et al 2018)"""

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
        self.values_support = possible_values_per_crit(self.train_set)
        # Set of the possible values in the train_set for each criterion
        self.alternatives_per_class = [[
            u for u in range(len(self.train_set)) if self.labels[u] == h
        ] for h in range(self.gen.num_classes)]

        # Generating triplet (i, h, k) and coalition set B as mentioned in
        # Section 3.4,Definition 4 (SAT encoding for U-NCS)
        self.variables = {
            "frontier_var":
            list({(i, h, k)
                  for i in range(self.gen.num_criteria)
                  for h in range(1, self.gen.num_classes)
                  for k in self.values_support[i]}),
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

        # Concatenates both encoding for the solver
        self.i2v = {}
        self.i2v.update(self.front_i2v)
        self.i2v.update(self.coal_i2v)

        # Results to be shared to predict
        self.frontier = {i: [0]*self.gen.num_criteria for i in range(1, self.gen.num_classes)}
        self.suff_coal = ()

        self.gopherpath = None

    def set_gophersat_path(self, gopherpath):
        self.gopherpath = gopherpath

    def clauses_2a(self) -> list:
        """Computes ascending scales clauses (named 2a in Definition 4)
        For all criteria i, classes h and adjacent pairs of value k<k'<k":
        x_{i, h, k} and x_{i, h, k"} => x_{i, h, k'}

        Returns:
            list: clauses according to the formula
        """
        clauses_2a = []
        # Only for adjacent values of k
        for i in range(self.gen.num_criteria):
            crit_values = self.values_support[
                i]  # Values are unique and already sorted
            for h in range(1, self.gen.num_classes):
                for ik in range(len(crit_values) - 2):
                    ikp = ik + 1
                    iks = ik + 2
                    clauses_2a.append([
                        self.front_v2i[(i, h, crit_values[ikp])],
                        -self.front_v2i[(i, h, crit_values[ik])],
                        -self.front_v2i[(i, h, crit_values[iks])]
                    ])

        return clauses_2a

    def clauses_2b(self) -> list:
        """Computes Hierarchy of profiles clauses (named 2b in Definition 4)
        (Evaluates classes (frontier) according to each value)
        For all criteria i, adjacent pairs of classes h<h', values k:
        x_{i, h', k} => x_{i, h, k}
        (being in higher classes makes the value greater to lower classes frontier)

        Returns:
            list: clauses according to the formula
        """
        # 3b Hierarchy of profiles
        clauses_2b = []
        # Only for adjacent values
        for i in range(self.gen.num_criteria):
            for h in range(1, self.gen.num_classes - 1):
                for k in set(self.values_support[i]):
                    clauses_2b.append([
                        self.front_v2i[(i, h, k)],
                        -self.front_v2i[(i, h + 1, k)]
                    ])
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
        # Only for a "adjacent" coalitions
        for B in self.coalitions:
            N_minus_B = {crit
                         for crit in range(self.gen.num_criteria)} - set(B)
            for i in N_minus_B:  # Adds exactly one element to the coalition
                Bp = set(B).union(set([i]))
                clauses_2c.append(
                    [self.coal_v2i[tuple(Bp)], -self.coal_v2i[B]])

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
                    clauses_2d.append([
                        -self.front_v2i[(i, h, self.train_set[u, i])]
                        for i in B
                    ] + [-self.coal_v2i[B]])

        return clauses_2d

    def clauses_2e(self) -> list:
        """Computes alternatives outranked by boundary bellow them clauses
        (named 2e in Definition 4)
        (Ensures the correct representation of the assignment (labels))
        For all coalition B, for all frontier h and all datapoint a assessed at class h
        (AND_{i in B} -x_{i, h, a_i}) => y_{N-B}
        (If the alternative is assigned to the right class
        whereas all of its values are below the frontier for the considered coalition
        then the complementary coalition is sufficient)

        Returns:
            list: clauses according to the formula
        """
        clauses_2e = []
        N = set(list(range(self.gen.num_criteria)))
        for B in self.coalitions:
            for h in range(1, self.gen.num_classes):
                for a in self.alternatives_per_class[h]:
                    clauses_2e.append([
                        self.front_v2i[(i, h, self.train_set[a, i])] for i in B
                    ] + [self.coal_v2i[tuple(N - set(B))]])
        return clauses_2e

    def run_solver(self) -> list:
        """Uses clasues defined above to encode the NCS problem
        into a SAT problem, solved by gophersat

        Returns:
            list: resulting frontiers between classes
        """
        # Add weights to the clauses
        hard_clauses = self.clauses_2a() + self.clauses_2b() + self.clauses_2c()
        soft_clauses = self.clauses_2d() + self.clauses_2e()
        hard_weight = len(soft_clauses) + 1
        
        for i, clause in enumerate(hard_clauses):
            hard_clauses[i] = [hard_weight] + clause
        
        for i, clause in enumerate(soft_clauses):
            soft_clauses[i] = [1] + clause           
        
        my_clauses = hard_clauses + soft_clauses
        
        my_dimacs = clauses_to_dimacs(
            my_clauses,
            len(self.variables["frontier_var"]) +
            len(self.variables["coalition_var"]), max_weight=hard_weight)

        write_dimacs_file(my_dimacs, "workingfile.wcnf")
        res = exec_gophersat("workingfile.wcnf", self.gopherpath)

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
            print("-              Unsatisfiable model, alternatives might be assigned to class 0              -")
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
                    crit = (None, None)
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
