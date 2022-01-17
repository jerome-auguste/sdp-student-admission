from time import time
from tools.generator import Generator
from tools.csvReader import csvReader
from mrsort import MRSort
from tools.parseArg import parseArguments
from tools.utils import print_comparison
from ncs import NcsSatModel


if __name__=='__main__':
    args = parseArguments()
    if args.file is None:
        gen = Generator(args.size, args.num_classes, args.num_criteria, args.lmbda, noisy=args.noisy, noise_percent= args.noise_percent)
        gen.display()
    else:
        rd = csvReader(args.file)
        gen = rd.to_generator()
        gen.display_imported()
    # gen = Generator(size=1000, num_classes=4, lmbda=0.5, weights=[0.2, 0.4, 0.25, 0.15], frontier=[12, 13, 10, 11])

    # MR_Sort
    mr_perf = {}
    mr_sort_begin = time()
    # print('\nMR-SORT')
    mrs = MRSort(gen)
    mrs.set_constraint()
    res = mrs.solve()
    mr_sort_end = time()

    mr_perf["time"] = mr_sort_end - mr_sort_begin
    mr_perf["train_pred"] = res
    mr_perf["test_pred"] = mrs.test()

    # NCS
    ncs_perf = {}
    # print("\nNCS")

    ncs_begin = time()

    u_ncs = NcsSatModel(generator=gen)
    u_ncs.set_gophersat_path(args.gopher_path)
    train_labels = u_ncs.train()
    ncs_end = time()
    test_labels = u_ncs.predict()

    ncs_perf["time"] = ncs_end - ncs_begin
    ncs_perf["train_pred"] = train_labels
    ncs_perf["test_pred"] = test_labels

    print_comparison(mr_perf=mr_perf, ncs_perf=ncs_perf, train_classes=gen.admission, test_classes=gen.admission_test)