#%%

from generator import Generator
from mrsort import MRSort

gen = Generator(size=1000, num_classes=3, lmbda=0.5, weights=[0.2, 0.4, 0.25, 0.15], frontier=[12, 13, 10, 11])
mrs = MRSort(gen)

mrs.print_data()
mrs.set_constraint()
mrs.solve()
mrs.print_res()