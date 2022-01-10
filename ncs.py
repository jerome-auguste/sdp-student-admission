import numpy as np
from itertools import product

from numpy.core.fromnumeric import size
from generator import Generator
from collections import Counter
import pprint as pp


train_set_size = 10
num_classes = 3
gen = Generator(size=train_set_size, num_classes=num_classes)

grades,admission = gen.generate()
print(f"Parameters:\nlambda: {gen.lmbda}\nweights: {gen.weights}\nfrontier: {gen.frontier}\nelements: {dict(Counter(admission))}\n")

# Boolean dimensions x_(i, h, k):
# - i is criterion (grades columns)
# - h is (ordered) class number (admission value)
# - k value taken for a criterion (grades value)

# y_B:
# - B coalition of criteria (B included in N)

# Generate the indexes of the students belonging to each class as a dict
# students_per_class = {}
# for cat in range(gen.num_classes):
#     students_per_class[cat] = []
#     for student in range (train_set_size):
#             if admission[student] == cat:
#                 students_per_class[cat].append(student)

# pp.pprint(students_per_class)

# Generate the indexes of the students per grade for each criterion
# students_per_grade = {}
# for i in range(gen.num_criterions):
#     students_per_grade[i] = {}
#     grades_list = [grades[student][i] for student in range(train_set_size)]
#     grades_set = set(grades_list)
#     for grade in grades_set:
#         students_per_grade[i][grade] = [s for s, g in enumerate(grades_list) if g == grade]

# pp.pprint(students_per_grade)
        

# The list format of the boolean value is i + student_index*num_criterias + 1 (in range(1,train_set_size*num_criterias))
# def squeezed_grades_index(student_index: int, i: int) -> int:
#     return i + student_index*gen.num_criterions + 1

# def squeezed_coalitions_index(coalition_index: int) -> int:
#     return train_set_size*gen.num_criterions + coalition_index

def grades_support_per_crit(grades: np.array):
    grades_set = []
    for i in range(gen.num_criterions):
        grades_set.append(sorted(list(set([grades[stud, i] for stud in range(grades.shape[0])]))))
    return grades_set

def subsets(criteria: list) -> list:
    if criteria == []:
        return [[]]
    x = subsets(criteria[1:])
    return x + [[criteria[0]] + y for y in x]

# Create variables/boolean value encoder
coalitions = [tuple(el) for el in subsets(list(range(gen.num_criterions)))]
grades_support = grades_support_per_crit(grades)

variables = {
    "frontier_var": [(i, h, k) for i in range(gen.num_criterions) for h in range(num_classes) for k in grades_support[i]],
    "coalition_var": coalitions
    }

front_v2i, front_i2v = dict(), dict()
front_v2i = {v: i+1 for i, v in enumerate(variables["frontier_var"])}
front_i2v = {i: v for v, i in front_v2i.items()}

coal_v2i, coal_i2v = dict(), dict()
coal_v2i = {v: i+len(front_i2v)+1 for i, v in enumerate(variables["coalition_var"])} # Indexes are starting right above where frontier indexing stops
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
#     for h in range(num_classes):
#         for ik in range(len(sorted_grades)):
#             for ikp in range(ik+1, len(sorted_grades[ik+1:])):
#                 if sorted_grades[ik] < sorted_grades[ikp]:
#                     clauses_3a.append([front_v2i[(i, h, sorted_grades[ikp])], -front_v2i[(i, h, sorted_grades[ik])]])
#                     # print(f"({i}, {h}, {sorted_grades[ikp]}) >  ({i}, {h}, {sorted_grades[ik]})")

# Only for adjacent values of k
for i in range(gen.num_criterions):
    sorted_grades = sorted(grades_support[i])
    for h in range(num_classes):
        for ik in range(len(sorted_grades)-1):
            ikp = ik+1
            while ikp < len(sorted_grades) and sorted_grades[ik] >= sorted_grades[ikp] :
                ikp += 1
            if ikp < len(sorted_grades):
                clauses_3a.append([front_v2i[(i, h, sorted_grades[ikp])], -front_v2i[(i, h, sorted_grades[ik])]])
                # print(f"({i}, {h}, {sorted_grades[ikp]}) >  ({i}, {h}, {sorted_grades[ik]})")

# 3b Hierarchy of profiles
clauses_3b = []

# Not only for adjacent values
# for i in range(gen.num_criterions):
#     for k in set(grades_support[i]):
#         for h in range(num_classes-1):
#             for hp in range(h+1, num_classes):
#                 clauses_3b.append([front_v2i[(i, h, k)], -front_v2i[(i, hp, k)]])
#                 # print(f"({i}, {h}, {k}) < ({i}, {hp}, {k})")

# Only for adjacent values
for i in range(gen.num_criterions):
    for k in set(grades_support[i]):
        for h in range(num_classes-1):
            clauses_3b.append([front_v2i[(i, h, k)], -front_v2i[(i, h+1, k)]])
            # print(f"({i}, {h}, {k}) < ({i}, {h+1}, {k})")

# 3c Coalitions strenghs


clause_3c = []

# Not only for "adjacent" coalitions (difference is a singleton)
for B in coalitions:
    for Bp in coalitions:
        if set(B).issubset(set(Bp)):
            clause_3c.append([coal_v2i[Bp], - coal_v2i[B]])
            # print(f"{B} is subset of {Bp}")
            
# Only for a "adjacent" coalitions
for B in coalitions:
    for i in range(gen.num_criterions):
        Bp = set(B).union(set([i]))
        if Bp != set(B):
            clause_3c.append([coal_v2i[tuple(Bp)], - coal_v2i[B]])
            # print(f"{B} is adjacent subset of {tuple(Bp)}")


# 3d Alternatives are outranked by boundary above them
students_per_class = [[u for u in range(train_set_size) if admission[u]==h] for h in range(num_classes)] # Students per category

clauses_3d = []
for B in coalitions:
    for h in range(1, num_classes):
        for u in students_per_class[h-1]:
            clauses_3d.append([-front_v2i[(i, h, grades[u, i])] for i in B ] + [-coal_v2i[B]])
            # print(f"-{[[(i, h, grades[u, i])] for i in B ]} OR -{B}")

# 3e Alternatives outrank the boundary bellow them
clauses_3e = []
for B in coalitions:
    for h in range(num_classes):
        for u in students_per_class[h]:
            N_minus_B = tuple(set(list(range(gen.num_criterions))) - set(B))
            clauses_3e.append([front_v2i[(i, h, grades[u, i])] for i in B ] + [coal_v2i[N_minus_B]])
            # print(f"{[[(i, h, grades[u, i])] for i in B ]} OR {N_minus_B}")


# Quelles sont les valeurs possibles pour les notes k ? Entier uniquement ?


# ------------------------------------------- TODO: adapt to the problem -------------------------------------------
#Construction du DIMCS et Résolution

import subprocess

def clauses_to_dimacs(clauses, numvar) :
    dimacs = 'c SAT NCS encoder \np cnf '+str(numvar)+' '+str(len(clauses))+'\n'
    for clause in clauses :
        for atom in clause :
            dimacs += str(atom) +' '
        dimacs += '0\n'
    return dimacs

def write_dimacs_file(dimacs, filename):
    with open(filename, "w", newline="") as cnf:
        cnf.write(dimacs)

#Attention à utiliser la vesion du solveur compatible avec votre système d'exploitation, mettre le solveur dans le même dossier que ce notebook        

def exec_gophersat(filename, cmd = "./gophersat.exe", encoding = "utf8") :
    result = subprocess.run([cmd, filename], stdout=subprocess.PIPE, check=True, encoding=encoding)
    string = str(result.stdout)
    lines = string.splitlines()

    if lines[1] != "s SATISFIABLE":
        return False, [], {}

    model = lines[2][2:].split(" ")

    return True, [int(x) for x in model if int(x) != 0], { i2v[abs(int(v))] : int(v) > 0 for v in model if int(v)!=0} 


myClauses = clauses_3a + clauses_3b + clause_3c + clauses_3d + clauses_3e
myDimacs = clauses_to_dimacs(myClauses, len(variables["frontier_var"]) + len(variables["coalition_var"]))

write_dimacs_file(myDimacs, "workingfile.cnf")
res = exec_gophersat("workingfile.cnf")

#Résultat
pp.pprint(res)