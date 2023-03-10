# SDP Student Admission

## WIP - [Report on overleaf](https://www.overleaf.com/9758323885jwgnydfhnmpb)

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
├── make_graph.py           # Script to generate the graph used in the report and presentation
├── single_peak_main.py     # Main script to be run in python env for single peak and maxsat problem 
├── single_peak_sat.py      # U-NCS single peak SAT model class
├── single_peak_maxsat.py   # U-NCS single peak and MaxSAT model class
├── MR-Sort-NCS.pdf         # Guidelines of the project
├── generate_csv.py         # Script to generate csv for graphs
├── tools                   # Tools and utilities
│   ├── generator.py        # Generates dataset based on parameters
│   ├── parseArg.py         # Command line argument parser used in main
│   ├── utils.py            # Utilities functions
│   └── csvReader.py        # Reader for csv data
│
├── requirements.txt    
├── LICENSE
├── .gitignore
└── README.md

```

## :mag: How to run the code

1. Install `requirements.txt` in your `Python` environment
2. Make sure you have a `gophersat` executable. Default name and location is `./gophersat.exe` (binary file in the root folder of the project). Please provide an alternative location with `--gopher-path` in others cases.
3. Run the command `python ./main.py [optionnal kwargs]`

Optionnal arguments you can pass in `main.py`:

- `-s` or `--size` for the size of the dataset (that will be splited into train and test sets)
- `-ncl` or `--num_classes` for the number of classes
- `-ncr` or `--num_criteria` for the number of criteria
- `-l` or `--lmbda` for the threshold value of the MR-Sort generator
- `-n` or `--noisy` to trigger noise on the dataset (set to 5%)
- `-npct` or `--noise_percent` to change percentage of noisy data (set to 5%)
- `-g` or `--gopher-path` to set the path to the GopherSat solver (default `./gophersat.exe`)
- `-f` or `--file` to set the path to a csv data file (will override the random generation which is the default behavior)

## :baby: Generator

Generates data according to the method described in [Leroy et al 2011](https://centralesupelec.edunao.com/pluginfile.php/214890/mod_label/intro/2011-Leroy-Mousseau-Pirlot-ADT.pdf) (see paragraph _Simulating an MR-Sort model n_).

- Initialize weights randomly in a normal distribution centered in 2 to avoid negative weight (redraws weights if one is negative), then normalize the weights vector.
- Initialize a random lambda in an uniform distribution in [0.5,1]
- Draws frontiers successively in an uniform distribution such that each frontier for each criterion is dominated by the next frontier.
- Then calculate the classes for each record.


## csvReader

Reads the data in a csv file. The data needs to be in the exact same format as provided (see this [file](https://centralesupelec.edunao.com/pluginfile.php/217659/mod_label/intro/data6crit50ex.csv))

## :1234: MR-Sort approach

Based on [Leroy et al 2011](https://centralesupelec.edunao.com/pluginfile.php/214890/mod_label/intro/2011-Leroy-Mousseau-Pirlot-ADT.pdf)

We want to maximize data separation. Therefore, for each student, accepted or rejected, we can define a slack compared to the threshold:

```math
\sum_{i}{w_{i}(s)}-\lambda - \sigma_s = 0 \forall s \in A^*
```

```math
\sum_{i}{w_{i}(s)}-\lambda + \sigma_s = 0 \forall s \in R^*
```

We will consider the margin between data from different classes (as done in SVM in Machine Learning), which is defined as the minimum slack:

```math
\alpha = \min_{s} \sigma_s
```

We want to maximize it, to maximize data separation:

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
- $`M\delta_i(s)+\epsilon \geq s_i - b_i \quad \forall s \in A^* \cup R^*, \forall i \in N`$
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
A state of the art SAT solver (gophersat) is then used to find a model for the solution, which is then decoded to be interpreted.

### Encoding the problem

We define 2 types of boolean variable to be used to model our problem:

- $`x_{i, h, k}`$ for each criterion $`i \in \mathcal{N}`$, for each boundary $`1 \leq h \leq p-1`$ and each grade $`k`$ of that criterion assuming the extreme classes undefined boundary are the extreme values taken by the grades
- $`y_B`$ for each coalition $`B \in \mathcal{P}(\mathcal{N})`$

Different clauses are then created, defining the rules our solver will respect:

- $`\forall i \in \mathcal{N}, \forall 1 \leq h \leq p-1, \forall k<k', \quad x_{i, h, k} \Rightarrow x_{i, h, k'}`$ (We can only consider adjacent grades) (1)
- $`\forall i \in \mathcal{N}, \forall 1 \leq h < h' \leq p-1, \forall k, \quad x_{i, h', k} \Rightarrow x_{i, h, k}`$ (we can only consider adjacent boundaries) (2)
- $`\forall B \subset B' \subseteq \mathcal{N}, \quad y_{B} \Rightarrow y_{B'}`$ (We can only consider $`B`$ and  $`B'`$ such that  $`|B' \setminus B| = 1`$) (3)
- $`\forall B \subseteq \mathcal{N}, \forall 1 \leq h \leq p-1 \forall u \in X^*: A(u) = C^{h-1}, \quad \bigwedge_{i \in B}{x_{i, h, u_i}} \Rightarrow \neg y_B`$ (4)
- $`\forall B \subseteq \mathcal{N}, \forall 1 \leq h \leq p-1 \forall a \in X^*: A(a) = C^{h}, \quad \bigwedge_{i \in B}{\neg x_{i, h, a_i}} \Rightarrow y_{\mathcal{N} \setminus B}`$  (5)


# :mountain: Single Peak problem

## U-NCS SAT approach

The formulation of the problem is similar, except for (1) which is replaced by :
- $`\forall i \in \mathcal{N}, \forall 1 \leq h \leq p-1, \forall k<k'<k'', \quad x_{i, h, k} \wedge x_{i,h,k''} \Rightarrow x_{i, h, k'}`$  (1')


## MaxSAT approach

As real life datasets might be noisy (misclassification of some alternatives), it is sometimes necessary to take that into account and find a way to correctly classify most of the alternatives. To do so, we can use a Weighted/Partial MaxSat approach (available with the same gophersat solver).

### Direct translation into MaxSAT

Our first model was to hierarchise our clauses, which is setting hard clauses and soft clauses as respectively rules that we don't want to infringe and rules that can be infringed (by noisy data).

That way, we want to keep the class, as well as the values and coalition hierarchies. Therefore, we have to set a hard clause $`w_{hard}`$ for clauses $`(1'), (2)`$ and $`(3)`$. However we want to relax the misclassification clauses for noisy alternatives, which means that we can set a weight of $`w_1`$ to clauses $`(4)`$ and $`(5)`$ so that $`w_{hard} > N_{soft} w_1`$

### Adapted formulation for MaxSAT

Using [Tlili et al. 2022](https://centralesupelec.edunao.com/pluginfile.php/209234/mod_label/intro/2022-Tili-et-al-EJOR.pdf), we had another approach using an adaptation of the clauses by adding a new boolean trigger $`z_x`$ that shows whether an alternative $`x \in X^*`$ is well classified by the model or not.
This leads to changing clauses $`(4)`$ and $`(5)`$ to the followings:

- $`\forall B \subseteq \mathcal{N}, \forall 1 \leq h \leq p-1 \forall u \in X^*: A(u) = C^{h-1}, \quad \bigwedge_{i \in B}{x_{i, h, u_i}} \Rightarrow \neg y_B \vee z_u`$ (4')
- $`\forall B \subseteq \mathcal{N}, \forall 1 \leq h \leq p-1 \forall a \in X^*: A(a) = C^{h}, \quad \bigwedge_{i \in B}{\neg x_{i, h, a_i}} \Rightarrow y_{\mathcal{N} \setminus B} \vee z_a`$  (5')

This changes gives more relaxation to the misclassification clauses: the clause can still be true in case of misclassification since the trigger can be set to `False`

We also had to add the clause that sets a soft rule for maximizing the number of well classified alternatives:

 ```math
 \forall x \in X^*, z_x
 ```

This approach can be found in the `maxsat` branch but since performances were not significantly better than the direct translation, this approach is still in experimental phase

As the formulation is quite different from the other one, we might need to properly decode the results of the MaxSAT solver to improve performance. This can be done for further contributions.
