# %%
import numpy as np
from random import uniform
from collections import Counter
from sklearn.model_selection import train_test_split


class Generator():
    FRONTIERS = ['all','valley','peak']
    def __init__(self,possible_frontiers = None, size: int = 100, num_classes: int = 2,
                 num_criteria: int = 4, lmbda: float = None,
                 weights: np.ndarray = None, frontier: np.ndarray = None,
                 size_test: float = 0.2, noisy: bool = False, noise_percent: float = 0.05) -> None:
        """
        Classe principale générant un dataset et les labels associés.
        La génération se fait a l'initialisation et stocke dans les attributs grades et labels les
        données générées (pour conserver les datasets pour la reproductibilité)
        KArgs :
            - size (int) : taille du dataset
            - size_test (float) : entre 0 et 1, pourcentage des données générées affectées au test set
            - noisy (bool) : si true, données perturbées en changeant aléatoirement la classe de certains élements dans le train set
            - noise_percent (float) : entre 0 et 1, pourcentage des données bruitées si l'option noisy est a true
            - num_classes (int) : nombre de classes a générer. Il y'aura num_classes-1 frontiers
            - num_criteria (int) : nombre de critères (ou notes par exemple). On tire ces notes selon une distrib uniforme entre 0 et 20
            - lmbda (float) : lambda définissant la "majorité" pour le modèle MR-Sort

        Attributs intéressants :
            - .grades : notes générées
            - .admission : labels générées
        """
        self.size_test = size_test
        self.size = size
        self.lmbda = lmbda
        self.num_classes = num_classes
        self.num_criteria = num_criteria
        self.possible_frontiers = possible_frontiers
        if possible_frontiers not in self.FRONTIERS:
            raise Exception(f'{possible_frontiers} not in {self.FRONTIERS}')
        if lmbda is None or lmbda > 1 or lmbda < 0:
            self.lmbda = uniform(0.5, 1)
        self.weights = weights
        if weights is None:
            self.weights = self.init_weights()
        self.frontier = frontier
        if frontier is None:
            if self.possible_frontiers is None:
                self.frontier = self.init_frontier()
            else:
                self.frontier = self.init_peak_frontier()
        self.grades, self.admission, self.grades_test, self.admission_test = self.generate(
            noisy=noisy, noise_percent=noise_percent)

    def init_weights(self) -> np.ndarray:
        # Genère les poids selon une distrib normale pour qu'ils ne soient pas trop différents
        w = np.random.standard_normal(self.num_criteria) + 2
        w /= w.sum()
        # On vérifie qu'il n'y a pas de poids négatif et on retire les poids tant que ça n'est pas le cas
        while any(w < 0):
            w = np.random.standard_normal(self.num_criteria) + 2
            w /= w.sum()
        return w

    def init_frontier(self) -> np.ndarray:
        # dernière limite basse (précédente frontière)
        last = np.zeros(self.num_criteria)
        frontiers = []
        for i in range(1, self.num_classes):
            # on tire une array de variable aléatoire sur une distrib uniforme entre la frontière précédente
            # de chaque critère et i/nombre de classes pour s'assurer de la dominance de la classe suivante sur la précédente,
            # comme dans l'article de référence
            last = np.random.uniform(
                last, [i*20/self.num_classes]*self.num_criteria)
            frontiers.append(last)
        return np.array(frontiers)

    def init_peak_frontier(self) -> np.ndarray:
        sizes = np.random.randint(1, 3, self.num_criteria)
        last = [np.zeros(sz) for sz in sizes if sz]
        for front in last:
            if len(front) > 1:
                front[1] = 20
        frontiers = []
        for cl in range(1, self.num_classes+1):
            arr = []
            for crit in range(self.num_criteria):
                if sizes[crit] == 1:
                    arr.append(np.random.uniform(
                        last[crit][0], cl*20/self.num_classes, sizes[crit]))
                else:
                    a = np.random.uniform(
                        last[crit][0], last[crit][1], sizes[crit])
                    a.sort()
                    arr.append(a)
            frontiers.append(arr)
            last = arr.copy()
        return frontiers

    def label_valley_frontier(self, grades: np.ndarray) -> np.ndarray:
        classes = np.zeros((self.size))
        for _, frontiers in enumerate(self.frontier):
            passed = np.zeros((self.size))
            for it, crit in enumerate(frontiers):
                if len(crit) == 1:
                    passed += (grades[:, it] > crit)*self.weights[it]
                else:
                    passed += ((grades[:, it] < crit[0]) *
                               (grades[:, it] > crit[1]))*self.weights[it]
            classes += passed > self.lmbda
        return classes

    def label_peak_frontier(self, grades: np.ndarray) -> np.ndarray:
        classes = np.zeros((self.size))
        for _, frontiers in enumerate(self.frontier):
            passed = np.zeros((self.size))
            for it, crit in enumerate(frontiers):
                if len(crit) == 1:
                    passed += (grades[:, it] > crit)*self.weights[it]
                else:
                    passed += ((grades[:, it] > crit[0]) *
                               (grades[:, it] < crit[1]))*self.weights[it]
            classes += passed > self.lmbda
        return classes

    def label_random_frontier(self, grades: np.ndarray) -> np.ndarray:
        classes = np.zeros((self.size))
        self.frontier_types = {i: np.random.choice(['valley', 'peak'])
                 for i, front in enumerate(self.frontier[0]) if front.size > 1}
        for _, frontiers in enumerate(self.frontier):
            passed = np.zeros((self.size))
            for it, crit in enumerate(frontiers):
                if len(crit) == 1:
                    passed += (grades[:, it] > crit)*self.weights[it]
                else:
                    if self.frontier_types[it] == 'valley':
                        passed += ((grades[:, it] < crit[0]) *
                                   (grades[:, it] > crit[1]))*self.weights[it]
                    if self.frontier_types[it] == 'peak':
                        passed += ((grades[:, it] > crit[0]) *
                                   (grades[:, it] < crit[1]))*self.weights[it]

            classes += passed > self.lmbda
        return classes

    def label(self, grades: np.ndarray) -> np.ndarray:
        passed = np.zeros((self.size))
        for frontier in self.frontier:
            # on itère sur les frontières :
            # grades > frontières renvoie une matrice, avec pour chaque ligne qui représente un élève, une array de
            # true ou false selon si la note est supérieure a la note correspondante de la frontier
            # on multiplie les booléens par les poids (false =  0, true = 1) puis on somme les poids obtenus par chaque élève,
            # et on compare a lambda : cela donne une nouvelle array (dim size) de booléens qu'on ajoute a passed.
            # Chaque fois qu'un élève passe une frontière, il passe dans la classe suivante. Comme les frontières sont
            # successivement dominées par construction, la génération se passe correctement.
            passed += ((grades > frontier) *
                       self.weights).sum(axis=1) > self.lmbda
        return passed

    def generate(self, noisy, noise_percent=0.05):
        # génère les notes et les labels
        grades = np.random.uniform(0, 20, (self.size, self.num_criteria))
        if self.possible_frontiers == 'peak':
            labels = self.label_peak_frontier(grades)
        elif self.possible_frontiers == 'valley':
            labels = self.label_valley_frontier(grades)
        elif self.possible_frontiers == 'all':
            labels = self.label_random_frontier(grades)
        else:
            labels = self.label(grades)
        grades, grades_test, labels, labels_test = train_test_split(
            grades, labels, test_size=self.size_test)
        if noisy:
            index_noisy = np.random.choice([i for i in range(
                len(labels))], size=np.random.randint(len(labels)*noise_percent))
            for index in index_noisy:
                labels[index] = np.random.randint(0, self.num_classes+1)
        return grades, labels, grades_test, labels_test

    def split(self, grades, labels):
        self.grades, self.grades_test, self.admission, self.admission_test = train_test_split(
            grades, labels, test_size=self.size_test)

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

    def display_imported(self):
        """
        Print the parameters used by the generator.
        """
        print(f"Parameters du generateur:\n",
              f"- size: {self.size}\n",
              f"- nombre de classes: {self.num_classes}\n",
              f"- nombre de critères: {self.num_criteria}\n",
              f"- echantillons par categorie: {dict(Counter(self.admission))}\n"
              )

# %%
a = Generator(peak=True)
# %%
