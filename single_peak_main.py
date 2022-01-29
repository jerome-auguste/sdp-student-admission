from time import time
from tools.generator import Generator
from tools.csvReader import csvReader
from tools.parseArg import parseArguments
from single_peak_sat import SinglePeakModel


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

    # Single Peak Model
    spm_perf = {}

    spm_begin = time()

    u_spm = SinglePeakModel(generator=gen)
    u_spm.set_gophersat_path(args.gopher_path)
    train_labels = u_spm.train()
    ncs_end = time()
    test_labels = u_spm.predict()

    spm_perf["time"] = ncs_end - spm_begin
    spm_perf["train_pred"] = train_labels
    spm_perf["test_pred"] = test_labels
    
    
    print(f"Duration: {spm_perf['time']}")
    print(f"Train perfs: {sum(spm_perf['train_pred'] == gen.admission)/len(spm_perf['train_pred'])}")
    print(f"Test perfs: {sum(spm_perf['test_pred'] == gen.admission_test)/len(spm_perf['test_pred'])}")