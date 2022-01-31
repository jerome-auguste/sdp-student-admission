from time import time
from tools.generator import Generator
from tools.csvReader import csvReader
from mrsort import MRSort
from tools.parseArg import parseArguments
from tools.utils import print_comparison
from ncs import NcsSatModel
import pandas as pd
from tools.utils import accuracy
from itertools import product

if __name__=='__main__':
    args = parseArguments()
    df = pd.DataFrame()
    for iter, size, noise, num_classes, num_criteria in product(range(5), range(25, 101, 25), range(0,16,5), range(2,5), range(3,7)):
        noisy = (noise > 0)
        gen = Generator(
            size, num_classes=num_classes, num_criteria=num_criteria, noisy=noisy, noise_percent=noise/100
        )

        # MR_Sort
        mr_perf = {}
        mr_sort_begin = time()
        mrs = MRSort(gen)
        mrs.set_constraint()
        res = mrs.solve()
        mr_sort_end = time()

        mr_perf["name"] = "MR-Sort"
        mr_perf["size"] = size
        mr_perf["noise"] = noise/100
        mr_perf["num_classes"] = num_classes
        mr_perf["num_criteria"] = num_criteria
        mr_perf["time"] = mr_sort_end - mr_sort_begin
        mr_perf["accuracy_on_train"] = accuracy(res, gen.admission)
        mr_perf["accuracy_on_test"] = accuracy(mrs.test(), gen.admission_test)
        df = df.append(mr_perf, ignore_index=True)

        # NCS
        ncs_perf = {}
        ncs_begin = time()

        u_ncs = NcsSatModel(generator=gen)
        u_ncs.set_gophersat_path(args.gopher_path)
        train_labels = u_ncs.train()
        ncs_end = time()
        test_labels = u_ncs.predict()

        ncs_perf["name"] = "NCS"
        ncs_perf["size"] = size
        ncs_perf["noise"] = noise/100
        ncs_perf["num_classes"] = num_classes
        ncs_perf["num_criteria"] = num_criteria
        ncs_perf["time"] = ncs_end - ncs_begin
        ncs_perf["accuracy_on_train"] = accuracy(train_labels, gen.admission)
        ncs_perf["accuracy_on_test"] = accuracy(test_labels, gen.admission_test)
        df = df.append(ncs_perf, ignore_index=True)
    df_mean = df.groupby(['name', 'size', 'noise', 'num_classes', 'num_criteria']).mean()
    df_mean.to_csv('results.csv')
