from generator import Generator
from mrsort import MRSort
from parseArg import parseArguments
from utils import print_res

if __name__=='__main__':
    args = parseArguments()
    gen = Generator(args.size, args.num_classes, args.num_criterions, args.lmbda, noisy=args.noisy)
    gen.display()
    # gen = Generator(size=1000, num_classes=4, lmbda=0.5, weights=[0.2, 0.4, 0.25, 0.15], frontier=[12, 13, 10, 11])
    
    # MR_Sort
    print('\nMR-SORT')
    mrs = MRSort(gen)
    mrs.set_constraint()
    res, compute_time = mrs.solve()
    print_res(
        compute_time,
        res, gen.admission,
        mrs.test(), gen.admission_test
    )

    # NCS
    print("\nNCS")
