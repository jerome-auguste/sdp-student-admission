#%%
import pandas as pd
from tools.generator import Generator
class csvReader():
    def __init__(self, path_to_csv):
        self.file = open(path_to_csv,'r').readlines()
        header = self.file[2].split(';')
        self.size = int(header[2])
        self.num_classes = int(header[1])
        self.num_criterions = int(header[0])
        lines = pd.DataFrame([line.replace('\n','').split(';') for line in self.file[3:]])
        self.grades = lines.iloc[:,1:self.num_criterions+1].to_numpy().astype(float)
        self.labels = lines.iloc[:,self.num_criterions+1].to_numpy().astype(float)

    def to_generator(self):
        gen = Generator()
        gen.size  = self.size
        gen.grades, gen.admission = self.grades, self.labels
        gen.num_classes, gen.num_criteria = self.num_classes, self.num_criterions
        gen.split(self.grades,self.labels)
        return gen
# %%
