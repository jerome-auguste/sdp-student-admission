"""Utils file with funtions used in the project"""

import numpy as np
import subprocess

def grades_support_per_crit(grade_record: np.ndarray) -> list:
    """Computes the (unique) existing grade values for each criterion

    Args:
        grade_record (np.ndarray): generated grades array (from Generator)

    Returns:
        list: ordered lists of unique grades for each criterion
    """
    grades_set = []
    for crit in range(len(grade_record[0])):
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