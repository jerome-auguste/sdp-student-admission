#%%
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()

    # Optional arguments

    parser.add_argument("-s", "--size", help="size.", type=int, default=100)
    parser.add_argument("-ncl", "--num_classes", help="Number of classes.", type=int, default=2)
    parser.add_argument("-ncr", "--num_criterions", help="Number of criterions.", type=int, default=4)
    parser.add_argument("-l", "--lmbda", help="Base lambda.", type=float, default=None)
    parser.add_argument("-n", "--noisy", help="Noise.", type=bool, default=False)


    return parser.parse_args()