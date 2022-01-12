#%%
import numpy as np
from random import uniform
from collections import Counter

class Generator():
    def __init__(self,size:int = 1000,num_classes:int= 2,num_criterions:int = 4, lmbda:float=None,weights:np.ndarray=None,frontier:np.ndarray=None) -> None:
        self.size = size
        self.lmbda = lmbda
        self.num_classes = num_classes
        self.num_criterions = num_criterions
        if lmbda is None:
            self.lmbda = uniform(0.5,1)
        self.weights = weights
        if weights is None:
            self.weights = self.init_weights()
        self.frontier = frontier
        if frontier is None:
            self.frontier = self.init_frontier()
        self.grades, self.admission = self.generate()

    def init_weights(self) -> np.ndarray:
        #Genère les poids selon une distrib normale pour qu'ils ne soient pas trop différents
        w = np.random.standard_normal(self.num_criterions) + 2
        w /= w.sum()
        #On vérifie qu'il n'y a pas de poids négatif et on retire les poids tant que ça n'est pas le cas
        while any(w < 0):
            w = np.random.standard_normal(self.num_criterions) + 2
            w /= w.sum()
        return w

    def init_frontier(self) -> np.ndarray:
        # dernière limite basse (précédente frontière)
        last = np.zeros(self.num_criterions)
        frontiers = []
        for i in range(1,self.num_classes):
            # on tire une array de variable aléatoire sur une distrib uniforme entre la frontière précédente
            # de chaque critère et i/nombre de classes pour s'assurer de la dominance de la classe suivante sur la précédente,
            # comme dans l'article de référence
            last = np.random.uniform(last,[i*20/self.num_classes]*self.num_criterions)
            frontiers.append(last)
        return np.array(frontiers)

    def label(self,grades:np.ndarray) -> np.ndarray:
        passed = np.zeros((self.size))
        for frontier in self.frontier:
            # on itère sur les frontières :
            # grades > frontières renvoie une matrice, avec pour chaque ligne qui représente un élève, une array de
            # true ou false selon si la note est supérieure a la note correspondante de la frontier
            # on multiplie les booléens par les poids (false =  0, true = 1) puis on somme les poids obtenus par chaque élève,
            # et on compare a lambda : cela donne une nouvelle array (dim size) de booléens qu'on ajoute a passed.
            # Chaque fois qu'un élève passe une frontière, il passe dans la classe suivante. Comme les frontières sont
            # successivement dominées par construction, la génération se passe correctement.
            passed += ((grades > frontier)*self.weights).sum(axis=1) > self.lmbda
        return passed

    def generate(self):
        # génère les notes et les labels
        grades = np.random.uniform(0,20,(self.size,self.num_criterions))
        return grades,self.label(grades)

    def display(self):
        print(f"Parameters du generateur:\n",
            f"- size: {self.size}\n",
            f"- lambda: {self.lmbda}\n",
            f"- weights: {self.weights}\n",
            f"- frontier: {self.frontier}\n",
            f"- echantillons par categorie: {dict(Counter(self.admission))}\n"
        )

# %%
