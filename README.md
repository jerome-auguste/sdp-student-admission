# SDP Student Admission

The project takes place in the context of the Systems Decision Process course of CentraleSupélec
The goal is to find the best parameters on an optimization problem to accept a set of students in the higher education program based on an assessment set (of different topics).

:date: Due date of the first part: **14/01/2022** \
:date: Due date of the entire project: **01/02/2022**

We want to maximize data separation. Therefore, for each student, we can define a (positive) loss compared to the threshold :

- $\sum_{i}{w_{i}(s)}-\lambda - \sigma_s = 0$ for $s \in A^*$
- $\sum_{i}{w_{i}(s)}-\lambda + \sigma_s = 0$ for $s \in R^*$

Let's define the *minimum error* over all students :

$$ \alpha = \min_{s} \sigma_s $$


LP :

$$ \max \alpha $$

s.t
$$
    \sum_{i \in N}{w_i(s)} + \sigma_s + \epsilon = \lambda \quad \forall s \in R^* \\
    
    \sum_{i \in N} w_i(s) = \lambda +  \sigma_s\quad\forall s \in A^* \\

    \alpha \leq \sigma_s\quad\forall s \in A^* \\

    w_i(s) \leq w_i \quad \forall s \in A^* \cup R^*, \forall i \in N \\

    w_i(s) \leq \delta_i(s) \quad \forall s \in A^* \cup R^*, \forall i \in N \\

    w_i(s) \geq \delta_i(s) - 1 + w_i \quad \forall s \in A^* \cup R^*, \forall i \in N \\

    M\delta_i(s)+\epsilon \geq s_i - b_i  \quad \forall s \in A^* \cup R^*, \forall i \in N \\

    M(\delta_i(s)-1) \leq s_i-b_i \quad \forall s \in A^* \cup R^*, \forall i \in N \\

    \sum_{i \in N}{w_i}=1, \quad \lambda \in [0.5, 1] \\
 
    w_i \in [0, 1], \forall i \in N \\

    w_i(s) \in [0, 1], \quad \delta_i(s) \in \{0, 1\} \quad \forall s \in A^* \cup R^*, \forall i \in N \\

    \sigma_s \in \R \quad \forall s \in A^* \cup R^* \\

    \alpha \in \R
$$
