#%%

from generator import Generator
from mrsort import MRSort

gen = Generator(size=1000)
mrs = MRSort(gen)

mrs.print_data()
mrs.set_constraint()
mrs.solve()
mrs.print_res()