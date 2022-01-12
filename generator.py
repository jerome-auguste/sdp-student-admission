#%%
import numpy as np
from random import uniform
from collections import Counter
from sklearn.model_selection import train_test_split

class Generator():
    def __init__(self,size:int = 100,num_classes:int= 2,num_criterions:int = 4, lmbda:float=None,weights:np.ndarray=None,frontier:np.ndarray=None, size_test:float = 0.2, noisy = False) -> None:
        """
        Classe principale générant un dataset et les labels associés.
        La génération se fait a l'initialisation et stocke dans les attributs grades et labels les
        données générées (pour conserver les datasets pour la reproductibilité)
        KArgs :
            - size (int) : taille du dataset
            - num_classes (int) : nombre de classes a générer. Il y'aura num_classes-1 frontiers
            - num_criterions (int) : nombre de critères (ou notes par exemple). On tire ces notes selon une distrib uniforme entre 0 et 20
            - lmbda (float) : lambda définissant la "majorité" pour le modèle MR-Sort

        Attributs intéressants :
            - .grades : notes générées
            - .admission : labels générées
        """
        self.size_test = size_test
        self.size = size
        self.lmbda = lmbda
        self.num_classes = num_classes
        self.num_criterions = num_criterions
        if lmbda is None or lmbda > 1 or lmbda < 0:
            self.lmbda = uniform(0.5,1)
        self.weights = weights
        if weights is None:
            self.weights = self.init_weights()
        self.frontier = frontier
        if frontier is None:
            self.frontier = self.init_frontier()
        self.grades, self.admission, self.grades_test, self.admission_test = self.generate(noisy=noisy)

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

    def generate(self, noisy):
        # génère les notes et les labels
        grades = np.random.uniform(0,20,(self.size,self.num_criterions))
        labels = self.label(grades)
        grades, grades_test, labels, labels_test = train_test_split(grades,labels,test_size=self.size_test)
        if noisy:
            index_noisy = np.random.choice([i for i in range(len(labels))],size=np.random.randint(len(labels)//3))
            for index in index_noisy:
                labels[index] = np.random.randint(0,self.num_classes+1)
        return grades, labels, grades_test, labels_test

    def display(self):
        """
        Print the parameters used by the generator.
        """
        print(f"Parameters du generateur:\n",
            f"- size: {self.size}\n",
            f"- lambda: {self.lmbda}\n",
            f"- weights: {self.weights}\n",
            f"- frontier: {self.frontier}\n",
            f"- echantillons par categorie: {dict(Counter(self.admission))}\n"
        )

# %%