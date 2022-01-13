from time import time
from generator import Generator
from mrsort import MRSort
from parseArg import parseArguments
from utils import print_comparison
from ncs import NcsSatModel


if __name__=='__main__':
    args = parseArguments()
    gen = Generator(args.size, args.num_classes, args.num_criterions, args.lmbda, noisy=args.noisy)
    gen.display()
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
    
    
    # print_res(
    #     mr_sort_end - mr_sort_begin,
    #     res, gen.admission,
    #     mrs.test(), gen.admission_test
    # )

    # NCS
    ncs_perf = {}
    # print("\nNCS")
    
    ncs_begin = time()
    
    u_ncs = NcsSatModel(generator=gen)
    train_labels = u_ncs.train()
    ncs_end = time()
    test_labels = u_ncs.predict()
    
    ncs_perf["time"] = ncs_end - ncs_begin
    ncs_perf["train_pred"] = train_labels
    ncs_perf["test_pred"] = test_labels
    
    print_comparison(mr_perf=mr_perf, ncs_perf=ncs_perf, train_classes=gen.admission, test_classes=gen.admission_test)