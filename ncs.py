"""This module solves an NCS problem with a SAT Solver (gophersat.exe)"""

import subprocess
from collections import Counter

import numpy as np

from generator import Generator

TRAIN_SET_SIZE = 50
NUM_CLASSES = 3
gen = Generator(size=TRAIN_SET_SIZE, num_classes=NUM_CLASSES)

grades, admission = gen.generate()
print(
    f"Parameters:\nfrontier: {gen.frontier}\nelements: {dict(Counter(admission))}\n"
)


def grades_support_per_crit(grade_record: np.array) -> list:
    """Computes the (unique) existing grade values for each criterion

    Args:
        grade_record (np.array): generated grades array (from Generator)

    Returns:
        list: ordered lists of unique grades for each criterion
    """
    grades_set = []
    for crit in range(gen.num_criterions):
        grades_set.append(
            sorted(
                list(set([grade_record[stud, crit]
                          for stud in range(grade_record.shape[0])]))))
    return grades_set


def subsets(criteria: list) -> list:
    """Generic function to generate all subsets of a subset

    Args:
        criteria (list): set of elements

    Returns:
        list: all subsets
    """
    if criteria == []:
        return [[]]
    subset = subsets(criteria[1:])
    return subset + [[criteria[0]] + y for y in subset]


# Create variables/boolean value encoder
coalitions = [tuple(el) for el in subsets(list(range(gen.num_criterions)))]
grades_support = grades_support_per_crit(grades)

variables = {
    "frontier_var": [(i, h, k) for i in range(gen.num_criterions)
                     for h in range(NUM_CLASSES) for k in grades_support[i]],
    "coalition_var":
    coalitions,
}

front_v2i, front_i2v = dict(), dict()
front_v2i = {v: i + 1 for i, v in enumerate(variables["frontier_var"])}
front_i2v = {i: v for v, i in front_v2i.items()}

coal_v2i, coal_i2v = dict(), dict()
coal_v2i = {
    v: i + len(front_i2v) + 1
    for i, v in enumerate(variables["coalition_var"])
}  # Indexes are starting right above where frontier indexing stops
coal_i2v = {i: v for v, i in coal_v2i.items()}

# Create the general index to value dictionnary
i2v = dict()
i2v.update(front_i2v)
i2v.update(coal_i2v)

# 3a Ascending scales
clauses_3a = []

# Not only adjacent values of k
# for i in range(gen.num_criterions):
#     sorted_grades = sorted(grades_support[i])
#     for h in range(NUM_CLASSES):
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
for i in range(gen.num_criterions):
    sorted_grades = sorted(grades_support[i])
    for h in range(NUM_CLASSES):
        for ik in range(len(sorted_grades) - 1):
            ikp = ik + 1
            while ikp < len(
                    sorted_grades) and sorted_grades[ik] >= sorted_grades[ikp]:
                ikp += 1
            if ikp < len(sorted_grades):
                clauses_3a.append([
                    front_v2i[(i, h, sorted_grades[ikp])],
                    -front_v2i[(i, h, sorted_grades[ik])],
                ])
                # print(f"({i}, {h}, {sorted_grades[ikp]}) >  ({i}, {h}, {sorted_grades[ik]})")

# 3b Hierarchy of profiles
clauses_3b = []

# Not only for adjacent values
# for i in range(gen.num_criterions):
#     for k in set(grades_support[i]):
#         for h in range(NUM_CLASSES-1):
#             for hp in range(h+1, NUM_CLASSES):
#                 clauses_3b.append([front_v2i[(i, h, k)], -front_v2i[(i, hp, k)]])
#                 # print(f"({i}, {h}, {k}) < ({i}, {hp}, {k})")

# Only for adjacent values
for i in range(gen.num_criterions):
    for k in set(grades_support[i]):
        for h in range(NUM_CLASSES - 1):
            clauses_3b.append(
                [front_v2i[(i, h, k)], -front_v2i[(i, h + 1, k)]])
            # print(f"({i}, {h}, {k}) < ({i}, {h+1}, {k})")

# 3c Coalitions strenghs

clause_3c = []

# Not only for "adjacent" coalitions (difference is a singleton)
for B in coalitions:
    for Bp in coalitions:
        if set(B).issubset(set(Bp)):
            clause_3c.append([coal_v2i[Bp], -coal_v2i[B]])
            # print(f"{B} is subset of {Bp}")

# Only for a "adjacent" coalitions
for B in coalitions:
    for i in range(gen.num_criterions):
        Bp = set(B).union(set([i]))
        if Bp != set(B):
            clause_3c.append([coal_v2i[tuple(Bp)], -coal_v2i[B]])
            # print(f"{B} is adjacent subset of {tuple(Bp)}")

# 3d Alternatives are outranked by boundary above them
students_per_class = [[u for u in range(TRAIN_SET_SIZE) if admission[u] == h]
                      for h in range(NUM_CLASSES)]  # Students per category

clauses_3d = []
for B in coalitions:
    for h in range(1, NUM_CLASSES):
        for u in students_per_class[h - 1]:
            clauses_3d.append([-front_v2i[(i, h, grades[u, i])]
                               for i in B] + [-coal_v2i[B]])
            # print(f"-{[[(i, h, grades[u, i])] for i in B ]} OR -{B}")

# 3e Alternatives outrank the boundary bellow them
clauses_3e = []
for B in coalitions:
    for h in range(NUM_CLASSES):
        for u in students_per_class[h]:
            N_minus_B = tuple(set(list(range(gen.num_criterions))) - set(B))
            clauses_3e.append([front_v2i[(i, h, grades[u, i])]
                               for i in B] + [coal_v2i[N_minus_B]])
            # print(f"{[[(i, h, grades[u, i])] for i in B ]} OR {N_minus_B}")

# Quelles sont les valeurs possibles pour les notes k ? Entier uniquement ?

# -------------------------------------- TODO: score results --------------------------------------
# Construction du DIMACS et Résolution
def clauses_to_dimacs(clauses: list, numvar: int) -> str:
    """Generates gophersat interpretable clauses (in cnf)

    Args:
        clauses (list): clauses to be parsed
        numvar (int): number of variable in the problem

    Returns:
        str: parsed clauses for gophersat
    """
    dimacs = ("c SAT encoded NCS problem \np cnf " + str(numvar) + " " +
              str(len(clauses)) + "\n")
    for clause in clauses:
        for atom in clause:
            dimacs += str(atom) + " "
        dimacs += "0\n"
    return dimacs


def write_dimacs_file(dimacs: str, filename: str):
    """Writes the generated string from clauses_to_dimacs function to a .cnf file

    Args:
        dimacs (str): generated string
        filename (str): file to save the string to
    """
    with open(filename, "w", newline="", encoding='utf8') as cnf:
        cnf.write(dimacs)


# Attention à utiliser la vesion du solveur compatible avec votre système d'exploitation,
# mettre le solveur dans le même dossier que ce notebook
def exec_gophersat(filename: str,
                   cmd: str = "./gophersat.exe",
                   encoding: str = "utf8") -> tuple[bool, list, dict]:
    """Executes gophersat on parsed text file (usually in .cnf)

    Args:
        filename (str): file to read clauses from
        cmd (str, optional): path to gophersat executable. Defaults to "./gophersat.exe".
        encoding (str, optional): encoding of the text file. Defaults to "utf8".

    Returns:
        tuple[bool, list, dict]: ("is it satisfiable",
                                    "model over index",
                                    "assigns to each variable a boolean value")
    """
    result = subprocess.run([cmd, filename],
                            stdout=subprocess.PIPE,
                            check=True,
                            encoding=encoding)
    string = str(result.stdout)
    lines = string.splitlines()

    if lines[1] != "s SATISFIABLE":
        return False, [], {}

    model = lines[2][2:].split(" ")

    return (
        True,
        [int(x) for x in model if int(x) != 0],
        {i2v[abs(int(v))]: int(v) > 0
         for v in model if int(v) != 0},
    )


MY_CLAUSES = clauses_3a + clauses_3b + clause_3c + clauses_3d + clauses_3e
MY_DIMACS = clauses_to_dimacs(
    MY_CLAUSES,
    len(variables["frontier_var"]) + len(variables["coalition_var"]))

write_dimacs_file(MY_DIMACS, "workingfile.cnf")
res = exec_gophersat("workingfile.cnf")

# Résultat
is_sat, i_model, var_model = res
front_results = [x for x in variables["frontier_var"] if var_model[x]]
coal_results = [x for x in variables["coalition_var"] if var_model[x]]

print(f"Resulted sufficient coalitions: {coal_results}")

frontier = []
for h in range(1, NUM_CLASSES):
    class_front = []
    for i in range(gen.num_criterions):
        crit_res = [x[2] for x in front_results if x[0] == i and x[1] == h]
        if len(crit_res) > 0:
            class_front.append(min(crit_res))
    frontier.append(class_front)

print(f"Resulted frontiers: {frontier}")
