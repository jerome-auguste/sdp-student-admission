"""Utils file with funtions used in the project"""

import subprocess
from collections import Counter
import numpy as np


def possible_values_per_crit(values_record: np.ndarray) -> list:
    """Computes the (unique) existing values for each criterion

    Args:
        values_record (np.ndarray): generated grades array (from Generator)

    Returns:
        list: sorted lists of unique grades for each criterion
    """
    values_set = []
    for crit in range(len(values_record[0])):
        values_set.append(
            sorted(
                list({values_record[stud, crit]
                        for stud in range(values_record.shape[0])})))
    return values_set

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

    return (True, model)


def print_res(compute_time, res_train, admission, res_test=None, admission_test=None):
    """
    Print results and accuracy of a solver.

    Args:
        admission: array of category for each sample (ground truth)
        compute_time: computing time spent
        res_train: array of category found by the solver for each sample of the train set
        res_data: array of category found by the solver for each sample of the test set
    """
    print(f"Resultats:\n",
        "- temps de calcul: {:.3f}s\n".format(compute_time),
        f"- Sur le train set:\n",
        f"  - elements par categorie: {dict(Counter(res_train))}\n",
        f"  - precision: {sum([res_train[i]==admission[i] for i in range(len(res_train))])/len(res_train)}"
    )
    if res_test is not None:
        print(f"- Sur le test set (generalisation):\n",
            f"  - elements par categorie: {dict(Counter(res_test))}\n",
            f"  - precision: {sum([res_test[i]==admission_test[i] for i in range(len(res_test))])/len(res_test)}\n"
        )
