# SDP Student Admission

## :computer: Contributors

- [Ruben Partouche](https://gitlab-student.centralesupelec.fr/2018partouchr)
- [Nathan Bruckmann](https://gitlab-student.centralesupelec.fr/2018bruckmann)
- [Jerome Auguste](https://gitlab-student.centralesupelec.fr/2018augustej)

## :page_facing_up: Context

The project takes place in the context of the Systems Decision Process course of CentraleSupélec
The goal is to find the best parameters on an optimization problem to accept a set of students in the higher education program based on an assessment set (of different topics/criteria).

:date: Due date of the first part: **14/01/2022** \
:date: Due date of the entire project: **01/02/2022**

## :file_folder: Project layout

```text
.
├── main.py                 # Main script to be run in python environment
├── mrsort.py               # MR-Sort model class
├── ncs.py                  # U-NCS SAT model class
├── MR-Sort-NCS.pdf         # Guidelines of the project
├── tools                   # Tools and utilities
│   ├── generator.py        # Generates dataset based on parameters
│   ├── parseArg.py         # Command line argument parser used in main
│   └── utils.py            # Utilities functions
│
├── LICENSE
├── .gitignore
└── README.md

```

## :mag: How to run the code

**TODO**: Update this section if the generator is changed (eg. for noise control)

1. Install `requirements.txt` in your `Python` environment
2. Place the `gophersat.exe` binary file in the root folder of the project
3. Run the command `python ./main.py [optionnal kwargs]`

Optionnal arguments you can pass in `main.py`:

- `-s` or `--size` for the size of the dataset (that will be splited into train and test sets)
- `-ncl` or `--num_classes` for the number of classes
- `-ncr` or `--num_criteria` for the number of criteria
- `-l` or `--lmbda` for the threshold value of the MR-Sort generator
- `-n` or `--noisy` to trigger noise on the dataset (set to 5%)

## :1234: MR-Sorst approach

**TODO**: Update with multi-classes formulas + other comments

We want to maximize data separation. Therefore, for each student, we can define a (positive) loss compared to the threshold and the *minimum loss* over the train set:

```math
\sum_{i}{w_{i}(s)}-\lambda - \sigma_s = 0 \forall s \in A^*
```

```math
\sum_{i}{w_{i}(s)}-\lambda + \sigma_s = 0 \forall s \in R^*
```

Considering the margin between data from different classes (as done in SVM in Machine Learning):

```math
\alpha = \min_{s} \sigma_s 
```

We want to maximize it

```math
\max \alpha 
```

s.t.

- $`\sum_{i \in N}{w_i(s)} + \sigma_s + \epsilon = \lambda \quad \forall s \in R^*`$
- $`\sum_{i \in N} w_i(s) = \lambda +  \sigma_s \quad \forall s \in A^*`$
- $`\alpha \leq \sigma_s \quad \forall s \in A^*`$
- $`w_i(s) \leq w_i \quad \forall s \in A^* \cup R^*, \forall i \in N`$
- $`w_i(s) \leq \delta_i(s) \quad \forall s \in A^* \cup R^*, \forall i \in N`$
- $`w_i(s) \geq \delta_i(s) - 1 + w_i \quad \forall s \in A^* \cup R^*, \forall i \in N`$
- $`M\delta_i(s)+\epsilon \geq s_i - b_i$  \quad $\forall s \in A^* \cup R^*, \forall i \in N`$
- $`M(\delta_i(s)-1) \leq s_i-b_i \quad \forall s \in A^* \cup R^*, \forall i \in N`$
- $`\sum_{i \in N}{w_i}=1, \quad \lambda \in [0.5, 1]`$
- $`w_i \in [0, 1] \quad \forall i \in N`$
- $`w_i(s) \in [0, 1], \quad \delta_i(s) \in \{0, 1\} \quad \forall s \in A^* \cup R^*, \forall i \in N`$
- $`\sigma_s \in \mathbb{R} \quad \forall s \in A^* \cup R^*`$
- $`\alpha \in \mathbb{R}`$

## :ballot_box_with_check: U-NCS SAT approach

### Concept

Based on [Belahcène et al 2018](https://centralesupelec.edunao.com/pluginfile.php/214890/mod_label/intro/2018-Belahcene-et-al-COR.pdf)

A completely different aproach (having better performance in computing time but more sensitive to noise) in which encode the problem into a SAT problem.

We define values for each class, each criteriion and each value (grade) of the training set.

We then define clauses that model the properties and the hypothesis that the training set should respect (eg. ordered classes $`C^1 \prec ... \prec C^p`$)
A state of the art sat solver (gophersat) is then used to find a model for the solution, which is then decoded to be interpreted.

### Encoding the problem

We define 2 types of boolean variable to be used to model our problem:

- $`x_{i, h, k}`$ for each criterion $`i \in \mathcal{N}`$, for each boundary $`1 \leq h \leq p-1`$ and each grade $`k`$ of that criterion assuming the extreme classes undefined boundary are the extreme values taken by the grades
- $`y_B`$ for each coalition $`B \in \mathcal{P}(\mathcal{N})`$

Different clauses are then created, defining the rules our solver will respect:

- $`\forall i \in \mathcal{N}, \forall 1 \leq h \leq p-1, \forall k<k', \quad x_{i, h, k} \Rightarrow x_{i, h, k'}`$ (We can only consider adjacent grades)
- $`\forall i \in \mathcal{N}, \forall 1 \leq h < h' \leq p-1, \forall k, \quad x_{i, h, k} \Rightarrow x_{i, h', k'}`$ (we can only consider adjacent boundaries)
- $`\forall B \subset B' \subseteq \mathcal{N}, \quad y_{B} \Rightarrow y_{B'}`$ (We can only consider $`B`$ and  $`B'`$ such that  $`|B' \setminus B| = 1`$)
- $`\forall B \subseteq \mathcal{N}, forall 1 \leq h \leq p-1 \forall u \in X^*: A(u) = C^{h-1}, \quad \bigwedge_{i \in B}{x_{i, h, u_i}} \Rightarrow \neg y_B`$
- $`\forall B \subseteq \mathcal{N}, forall 1 \leq h \leq p-1 \forall a \in X^*: A(a) = C^{h-1}, \quad \bigwedge_{i \in B}{\neg x_{i, h, a_i}} \Rightarrow y_{\mathcal{N} \setminus B}`$
