#%%
import numpy as np

SCALE = 20

class Generator():
    def __init__(self,size:int,lmbda:float,weights:np.ndarray,frontier:np.ndarray,scale=SCALE) -> None:
        self.size = size
        self.num_criterions = len(weights)
        self.weights = weights
        self.frontier = frontier
        self.scale = scale
        self.lmbda = lmbda

    def label(self,grades:np.ndarray) -> np.ndarray:
        ok = grades > self.frontier
        return (ok*self.weights).sum() > self.lmbda

    def generate(self):
        grades = np.random.rand(self.size,self.num_criterions)*self.scale
        labels = self.label(grades)
        return grades,labels
# %%
