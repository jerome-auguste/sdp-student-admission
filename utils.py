    """utils file with funtions used in the project
    """

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