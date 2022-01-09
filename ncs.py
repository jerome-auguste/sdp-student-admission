import numpy as np
from itertools import product

from numpy.core.fromnumeric import size
from pysat.solvers import Glucose3
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
students_per_class = {}
for cat in range(gen.num_classes):
    students_per_class[cat] = []
    for student in range (train_set_size):
            if admission[student] == cat:
                students_per_class[cat].append(student)

pp.pprint(students_per_class)

# Generate the indexes of the students per grade for each criterion
students_per_grade = {}
for i in range(gen.num_criterions):
    students_per_grade[i] = {}
    grades_list = [grades[student][i] for student in range(train_set_size)]
    grades_set = set(grades_list)
    for grade in grades_set:
        students_per_grade[i][grade] = [s for s, g in enumerate(grades_list) if g == grade]

pp.pprint(students_per_grade)
        

# The list format of the boolean value is i + student_index*num_criterias + 1 (in range(1,train_set_size*num_criterias))
def squeeze_grades(student_index: int, i: int) -> int:
    return i + student_index*gen.num_criterions + 1

def squeeze_coalitions(coalition_index: int) -> int:
    return train_set_size*gen.num_criterions + coalition_index

g = Glucose3()

# 3a
for i in range(gen.num_criterions):
    for h in range(num_classes):
        for stud1 in students_per_class[h]:
            for stud2 in students_per_class[h]:
                if (grades[stud1][i] < grades[stud2][i]):
                    g.add_clause([squeeze_grades(stud1, i), -squeeze_grades(stud2, i)])
    
# 3b
for i in range(gen.num_criterions):
    for k in students_per_grade[i].keys():
        for stud1 in students_per_grade[i][k]:
            for stud2 in students_per_grade[i][k]:
                if (admission[stud1] < admission[stud2]):
                    g.add_clause([squeeze_grades(stud1, i), -squeeze_grades(stud2, i)])

# 3c
def subsets(criteria: list) -> list:
    if criteria == []:
        return [[]]
    x = subsets(criteria[1:])
    return x + [[criteria[0]] + y for y in x]

coalitions = subsets(list(range(gen.num_criterions)))

for ib in range(len(coalitions)):
    for ibp in range(len(coalitions)):
        if coalitions[ib].issubset(coalitions[ibp]):
            g.add_clause(squeeze_coalitions(ibp), - squeeze_coalitions(ib))


# 3d
for ib in range(len(coalitions)):
    for h in range(num_classes):
        pass
# Quelle est la différence entre h et A(u) ? Classe/frontière estimée vs frontière évaluée ? Comment définir la frontière estimée ?
# Quelles sont les valeurs possibles pour les notes k ? Entier uniquement ?

# g = Glucose3()
# g.add_clause([-1, 2])
# g.add_clause([-2, 3])
# print(g.solve())
# print(g.get_model())

