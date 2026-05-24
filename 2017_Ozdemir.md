# Recursive Tensor Subspace Tracking for Dynamic Brain Network Analysis

**Alp Ozdemir, Edward M. Bernat, and Selin Aviyente**

*IEEE Transactions on Signal and Information Processing over Networks, Vol. 3, No. 4, December 2017*

---

## Abstract

Recent years have seen a rapid growth in computational methods for a better understanding of functional connectivity brain networks constructed from neuroimaging data. Most of the current work has been limited to static functional connectivity networks (FCNs), where the relationships between different brain regions is assumed to be stationary. Recent work indicates that functional connectivity is a dynamic process over multiple time scales and the dynamic formation and dissolution of connections plays a key role in cognition, memory, and learning. In the proposed work, we introduce a tensor-based approach for tracking dynamic functional connectivity networks. The proposed framework introduces a robust low-rank+sparse structure learning algorithm for tensors to separate the low-rank community structure of connectivity networks from sparse outliers. The proposed framework is used to both identify change points, where the low-rank community structure of the FCN changes significantly, and summarize this community structure within each time interval. The proposed framework is applied to the study of cognitive control from electroencephalogram data during a Flanker task.

**Index Terms:** Dynamic functional connectivity networks, electroencephalogram, low-rank tensor decomposition.

---

## I. Introduction

Advanced functional imaging techniques such as EEG and functional magnetic resonance imaging (fMRI) have enabled the study of the neuronal mechanisms underlying cognition in detail. These studies have revealed that transient synchronization, referred to as functional connectivity (FC), between spatially distributed neural populations is responsible for human cognition, perception and emotion. Recently, tools from graph theory have been employed to analyze the functional connectivity of the brain by associating nodes with distinct brain regions and edges with pairwise interactions between them. Most of the current work on functional connectivity network analysis focuses on static networks where the networks correspond to average activity over a time and frequency window of interest. However, recent studies have shown that functional connectivity networks change dynamically in short time scales and exhibit task-related patterns. This continuous formation and destruction of functional connectivity also controls the emergence of a unified neural process in cognition, perception and memory.

To better understand the brain dynamics, early studies focused on extracting graph theoretic measures across time for a time-varying analysis of FC graphs. For example, Valencia et al. showed how the small-world structure of FC networks evolve during a visual stimulus. Similarly, Fallani et al. presented a graph theoretical approach for FC networks to identify persistent edges during a motor task. Recently, dynamic FC network (dFCN) tracking approaches have been combined with network state estimation techniques. Allen et al. assume that the FC network at each time point is at a distinct network state where the network states are determined through k-means clustering of dFCNs across time and subjects from resting state fMRI data. Similarly, Dimitriadis et al. introduced FC microstates inspired by the EEG microstate literature. In, the network states are obtained through clustering and Markov modelling to identify both FC-states and the transitions between them. Similarly, the network states are identified by evolutionary clustering applied to FC edge timeseries. An alternative approach to dFCN analysis assumes that network states are made up of multiple building blocks. Leonardi et al. propose a principal component analysis (PCA) based approach to reveal the intrinsic FC patterns named as eigenconnectivities, and describe the FC matrices as weighted sum of eigenconnectivities. Similarly, different PCA based approaches have been used to identify dynamics of both resting-state and task-based EEG. More recently, clustering and SVD based approaches were compared to identify task related and resting state FC dynamics, showing that FC patterns obtained from an SVD based approach better represent the task-based dynamics, while patterns obtained from a clustering based approach are more suitable to identify resting state FC dynamics.

In addition to approaches focused on unraveling the network states from dFCNs, recently introduced methods have also focused on detection of change points where the network structure considerably changes. Cribben et al. presented a data-driven technique which first detects the temporal change points by a greedy partitioning scheme and then estimates a connectivity graph for FC patterns in each temporal partition. Zhang et al. presented dynamic Bayesian variable partition model which simultaneously identifies the state transitions and learns significant FC patterns in each state. Similarly, Ou et al. proposed a Bayesian model to detect change points from functional brain interactions and evaluated the network structure using nonnegative matrix factorization within each temporal segment. However, all of these approaches make use of time-series data and are not based on the dFCNs constructed directly from data. Moreover, these approaches assume a multivariate Gaussian model for the underlying time series data and require individual analysis of each subject before inferring the group's network structure. Finally, these methods are computationally expensive as they rely either on greedy search or probabilistic metrics.

In this paper, we propose a tensor based representation of dFCNs for tracking and summarizing the functional connectivity across time and subjects from task-based EEG data. First, we introduce a tensor subspace analysis method for robust low-rank + sparse structure recovery. This is motivated by the fact that FCNs are known to have a modular structure which translates to a low-rank connectivity matrix. In the case of dFCNs, these low-rank structures can be assumed to change slowly in time, similar to EEG microstates, which can be described as a slowly changing subspace. Conventional subspace analysis methods such as PCA and SVD cannot deal with higher order data and existing tensor decomposition methods such as higher-order SVD (HoSVD) and parallel factor analysis (PARAFAC) are not robust to sparse outliers or noise in the data. The proposed recursive framework separates the low-rank part of the data from sparse noise components by identifying change points and updating the estimates of low-rank subspaces. Identified change points corresponding to the subspace change along the connectivity mode of the tensor are used to define time intervals of interest. The low-rank tensor within each time interval is then summarized through a recently introduced multiple network clustering approach known as Fiedler Consensus Clustering Approach (FCCA). Finally, the proposed framework is applied to dFCNs constructed from EEG data collected during a study of error-related negativity.

The proposed framework offers three major contributions to both the literature in online tensor subspace tracking and dFCN analysis. First, unlike most of the current work which focuses on the dynamics of resting state fMRI networks, we focus on the dynamics of task related networks from EEG data. As EEG data has high temporal resolution and the data considered in this paper is response locked, determining the actual time points where significant changes to network structure occurs is highly relevant. The proposed framework offers a way to determine time intervals during which the FCN has a common quasi-stationary pattern across time and subjects, i.e. slowly changing subspace structure similar to microstates. Second, most of the current work reduces the high dimensionality of the dFCNs by vectorizing the connectivity matrices into long vectors before identifying the network states. This approach does not preserve the topological structure of the network. In the proposed work, we address this problem by keeping the network structure of FCNs intact by using tensor representations. Through tensor representation, we can capture the variability common to all subjects across time. Finally, the proposed low-rank plus sparse structure learning algorithm for tensors offers a novel way of recovering a low-rank subspace estimate along each mode of the data where the rank is defined through the Tucker rank. This rank definition is directly related to the modular structure of FCNs. The proposed approach separates the low-rank part of the data from sparse noise components along time and then summarizes the network within each time interval through clustering the extracted low-rank networks within the time interval. This yields better structure information as it is equivalent to denoising the networks.

---

## II. Background

### A. Robust Principal Component Analysis

High dimensional data mostly lies in a lower dimensional subspace and principal component analysis (PCA) is the most widely used technique to identify this lower dimensional subspace. Recently, PCA has been used for identifying network states from dynamic functional connectivity networks. However, it is known that PCA suffers from non-Gaussian corruptions and may find a completely wrong principal subspace in the presence of even a few outliers. These drawbacks have forced researchers to develop more robust subspace estimation techniques which is a significantly more difficult problem than standard PCA.

Since the recent work by Candes et al. and Chandrasekharan et al., the general problem of separating a sparse matrix and a low-rank matrix from their sum has received a lot of attention. The final goal usually is to either find the column span of the low-rank matrix or the support of the sparse one. This is now commonly referred to as the "low-rank + sparse recovery" problem. There has been a large amount of recent work on batch methods for low-rank + sparse recovery and its various extensions including Principal Component Pursuit, Outlier Pursuit and Low-Leverage Decomposition.

One of the well-known robust PCA methods is principal component pursuit (PCP) which assumes that the data matrix **M** has a low-rank part **L** and a sparse noise or outlier **S** as:

$$M = L + S \tag{1}$$

It was shown that **L** can be efficiently estimated by solving the following optimization problem:

$$\min \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad L + S = M \tag{2}$$

where $\lambda$ is the regularization parameter and $\|L\|_* = \sum_{i=1}^{r} \sigma_i(L)$ denotes the nuclear norm where $\sigma_i$'s are the first $r$ singular values of the matrix **L**. This problem has been solved by using convex optimization approaches, i.e., augmented Lagrange multiplier algorithm, accelerated proximal gradient approach.

In order to reduce the computational complexity and to achieve online subspace tracking, various approaches have also been proposed to solve the RPCA problem, i.e., GRASTA, PETRELS and REPROCS. These approaches first identify the subspace that the low rank data lies in, then recovers incoming low-rank measurement vectors from missing entries by considering this subspace information.

A recently introduced algorithm REPROCS recursively separates the low-rank part from sparse noise as follows. Let $M_t \in \mathbb{R}^{n \times 1}$ be a time-series of measurement vectors written as $M_t = L_t + S_t$ where $L_t$ is the low-rank part which lies in a subspace spanned by $P_t$ and $S_t$ is the sparse noise vector. Let $\hat{P}_t$ be an accurate estimate of the $r$-dimensional basis $P_t$ at time $t$ and $\hat{P}_{t,\perp}$ be the orthogonal complement of $\hat{P}_t$. Let $\alpha_t := \hat{P}_t^\top L_t$ be the projection of $L_t$ onto $\hat{P}_t$ and $\beta_t := (\hat{P}_{t,\perp})^\top L_t$ be a projection of $L_t$ onto $\hat{P}_{t,\perp}$. Then, $M_t$ can be rewritten as $M_t = \hat{P}_t \alpha_t + \hat{P}_{t,\perp} \beta_t + S_t$. REPROCS first projects the measurement vector onto $\hat{P}_{t,\perp}$ to approximately nullify the low-rank part $L_t$. As $y_t := (\hat{P}_{t,\perp}^\top) M_t$, where $y_t$ can be rewritten as $y_t = (\hat{P}_{t,\perp}^\top) S_t + \beta_t$, and the dimension of the projected data vector reduces to $n - r$. Since projecting $M_t$ onto $\hat{P}_{t,\perp}$ nullifies the contribution of $L_t$, $\beta_t$ can be interpreted as small noise. Therefore, solving for $n$-dimensional $S_t$ from $(n-r)$-dimensional $y_t$ becomes a traditional sparse recovery problem. Once $\hat{S}_t$ is recovered, $L_t$ can be estimated as $\hat{L}_t = M_t - \hat{S}_t$. Performance of this algorithm highly depends on the correctness of the estimated low-rank subspace and the slowly changing subspace assumption. However, this method is limited to vector type measurements, and cannot be applied directly to higher order datasets such as tensors. In this paper, we will present an extension of REPROCS to tensor type data.

Recently, tensor-based approaches have been proposed to track dynamic tensor subspaces such as dynamic tensor analysis, streaming tensor analysis and window based tensor analysis. However, these approaches provide computationally efficient frameworks for analysis of streaming datasets by recursively updating subspace information and do not address the robustness of the subspace estimates. Goldfarb and Qin extended robust PCA to tensors (HoRPCA) by solving low-rank + sparse recovery problem for general higher order tensors. However, this method is highly computationally expensive and does not update the subspaces online, i.e., would not be useful for tracking subspace changes across time. Li et al. presented a robust subspace learning algorithm (RTSL) that incrementally updates the tensor subspace. Moreover, Nion et al. proposed two adaptive approaches to track PARAFAC decomposition of 3-way tensors. These approaches suggest to update the PARAFAC decomposition at every time point based on simultaneous diagonalization or minimization of weighted least squares criterion. More recently, Mardani et al. proposed an online subspace learning method based on nuclear norm minimization and extended this approach for matrices and higher order datasets. Extension of this algorithm to tensors takes advantage of PARAFAC model to minimize tensor rank and considers temporal information as one of the tensor modes. Similar to that work, OLSTEC also tracks the subspace of partially observed higher-order data using PARAFAC decomposition.

### B. Tensor Algebra & Tensor Decompositions

An order $N$ tensor is denoted as $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ where $x_{i_1, i_2, \ldots, i_N}$ corresponds to the $(i_1, i_2, \ldots, i_N)$th element of the tensor $\mathcal{X}$. Vectors obtained by fixing all indices of the tensor except the one that corresponds to $n$th mode are called mode-$n$ fibers.

**Mode-$n$ product:** The mode-$n$ product of a tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots \times I_n \times \cdots \times I_N}$ and a matrix $U \in \mathbb{R}^{J \times I_n}$ is denoted as $\mathcal{Y} = \mathcal{X} \times_n U$ and is of size $I_1 \times \cdots \times I_{n-1} \times J \times I_{n+1} \times \cdots \times I_N$.

**Tensor matricization:** Process of reordering the elements of the tensor into a matrix is known as matricization or unfolding. The mode-$n$ matricization of tensor $\mathcal{Y} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ is denoted as $Y_{(n)} \in \mathbb{R}^{I_n \times \prod_{i \neq n} I_i}$ and is obtained by arranging mode-$n$ fibers to be the columns of the resulting matrix. Unfolding the tensor $\mathcal{Y} = \mathcal{X} \times_1 U_1 \times_2 U_2 \cdots \times_N U_N$ along mode-$n$ is equivalent to $Y_{(n)} = U_n X_{(n)} (U_N \otimes \cdots U_{n+1} \otimes U_{n-1} \cdots \otimes U_1)$, where $\otimes$ is the matrix Kronecker product.

**The $n$-Rank:** Let $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ be an $N$-way tensor, the $n$-rank of $\mathcal{X}$ is the collection of rank of mode matrices $X_{(n)}$ and is denoted as:

$$\text{rank}_n(\mathcal{X}) = \left(\text{rank}(X_{(1)}), \text{rank}(X_{(2)}), \ldots, \text{rank}(X_{(n)})\right) \tag{3}$$

where $n = 1, 2, \ldots, N$.

**Tucker decomposition:** Tucker decomposition is a form of higher order SVD. Any tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ can be decomposed as mode products of a core tensor $\mathcal{S} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ and $N$ mode matrices $U^{(n)} \in \mathbb{R}^{I_n \times I_n}$:

$$\mathcal{X} = \mathcal{S} \times_1 U^{(1)} \times_2 U^{(2)} \cdots \times_N U^{(N)} \tag{4}$$

where the matrix $U^{(n)}$ contains the left singular vectors of $X_{(n)}$ and $\mathcal{S}$ is obtained by $\mathcal{S} = \mathcal{X} \times_1 U^{(1)\top} \times_2 U^{(2)\top} \cdots \times_N U^{(N)\top}$.

### C. Time-Varying Measure of Phase Synchrony

In this paper, pairwise functional connectivity will be quantified using a recently introduced time-frequency phase estimation method based on Reduced Interference Rihaczek distribution (RID-Rihaczek). Phase synchronization within different frequency bands across the brain has been shown to be a plausible mechanism explaining neuronal integration.

Reduced Interference Rihaczek Distribution (RID-Rihaczek) is given by:

$$C(t, \omega) = \int\!\!\int \exp\!\left(-\frac{(\theta\tau)^2}{\sigma}\right) \exp\!\left(j\frac{\theta\tau}{2}\right) A(\theta, \tau) \, e^{-j(\theta t + \tau\omega)} \, d\tau \, d\theta \tag{5}$$

where $\exp(-(\theta\tau)^2/\sigma)$ is the Choi-Williams kernel used to filter out the cross-terms, $A(\theta, \tau) = \int x(u + \frac{\tau}{2}) x^*(u - \frac{\tau}{2}) e^{j\theta u} du$ is the ambiguity function of the signal $x(t)$ and $\exp(j\theta\tau/2)$ is the kernel corresponding to the Rihaczek distribution. The phase difference between two signals, $x_i$ and $x_j$, based on this complex distribution is computed as:

$$\Phi_{ij}(t, \omega) = \arg\left\{\frac{C_i(t,\omega) C_j^*(t,\omega)}{|C_i(t,\omega)||C_j(t,\omega)|}\right\} \tag{6}$$

where $C_i(t,\omega)$ and $C_j(t,\omega)$ refer to the complex energy distributions of the two signals $x_i(t)$ and $x_j(t)$ respectively. A synchrony measure quantifying the intertrial variability of the phase differences, phase locking value (PLV), is defined as:

$$\text{PLV}_{i,j}(t,\omega) = \frac{1}{\kappa} \left| \sum_{k=1}^{\kappa} \exp(j\Phi^k_{ij}(t,\omega)) \right| \tag{7}$$

where $\kappa$ is the number of trials and $\Phi^k_{ij}(t,\omega)$ is the time-varying phase estimate between two signals recorded at electrodes $i$ and $j$ for the $k$th trial. If the phase difference varies little across the trials, PLV is close to 1. Compared to the existing synchrony measures, RID-Rihaczek based phase synchrony measure is more robust to noise, and has uniformly high time-frequency resolution with less bias.

In this paper, we construct the functional connectivity matrices at each time point and for each subject as:

$$G_{i,j}(t) = \frac{1}{\omega_b - \omega_a} \sum_{\omega=\omega_a}^{\omega_b} \text{PLV}_{i,j}(t,\omega) \tag{8}$$

where the entries of the connectivity networks are computed as the average synchrony between pairs of nodes at time $t$ averaged over a frequency band of interest which is the theta band. Once the individual connectivity matrices are constructed, a three-way tensor $\mathcal{X}_t \in \mathbb{R}^{N \times N \times S}$ is formed at each time point across all subjects as:

$$\mathcal{X}_t(i,j,s) = G^s_{i,j}(t) \tag{9}$$

where $i, j \in \{1, 2, \ldots, N\}$ correspond to the nodes or brain regions in the network and $s \in \{1, 2, \ldots, S\}$ is the subject.

### D. Consensus Clustering and Fiedler Consensus Clustering Approach

In neuroscience problems, it is desirable to find a common network structure across subjects performing the same task or in the same population. Recently, we have introduced Fiedler Consensus Clustering Algorithm (FCCA) to address this issue to obtain a common community structure across multiple weighted graphs, where the graphs correspond to individual functional connectivity networks discussed in Section II-C. One of the most common ways to partition a graph is spectral clustering. Spectral clustering generally uses the eigenvectors of the Laplacian matrix computed as $L = D - A$, where $A$ is the adjacency matrix of the graph and $D$ is the degree matrix containing degrees of nodes along the diagonal with $D(i,i) = \sum_{j=1, j \neq i}^{N} A(i,j)$. The eigenvector corresponding to the second smallest non-zero eigenvalue of the Laplacian matrix provides the optimal minimal cut of a graph and this eigenvector is referred to as the Fiedler vector.

In FCCA, the original connectivity matrices are bi-partitioned into two clusters using the Fiedler partitioning method. This results in a cluster matrix for the $r$th network $T_r$ such that:

$$T_r(i,j) = \begin{cases} 1 & \text{if nodes } v_i, v_j \text{ are in the same cluster} \\ 0 & \text{otherwise} \end{cases} \tag{10}$$

and $r = \{1, 2, \ldots, m\}$ where $m$ is the number of networks.

In order to find the common community structure across multiple graphs, we introduce a co-occurence matrix $W$ where $W(i,j) = \frac{\sum_{r=1}^{m} T_r(i,j)}{m}$ and $W(i,j) \in [0,1]$. $W(i,j)$ is the probability that a pair of nodes are members of the same cluster across multiple graphs. The adjacency matrix reflects the strength of a direct relationship between a node pair, whereas $W$ reflects the likeliness that a pair of nodes are in the same cluster across all subjects.

The Laplacian matrix of $W$ is computed and the Fiedler vector is found to form a bi-partition of $W$ into a community structure composed of clusters $c_1$ and $c_{-1}$. Since $W$ represents the probability that a node pair should be clustered together, the Fiedler partition of $W$ represents the community structure common to all graphs. The initial partition set, $C = \{c_1, c_{-1}\}$, contains 2 clusters but if $k > 2$ is desired, the process can be repeated by selecting a cluster in $C$ to partition. In this case, $c_1$ or $c_{-1}$ is selected based on the quality score of each cluster. Next, sub-matrices are extracted from the original connectivity matrices such that they only contain the nodes of the chosen cluster. These sub-matrices are used to derive the new sub co-occurence matrix, $W_y$. The Fiedler partitioning is performed on the selected cluster to obtain two new clusters, $c^y_1$ and $c^y_{-1}$. The final cluster set $C = \{c_{-y}, c^y_1, c^y_{-1}\}$ is a concatenation of the two new clusters with the original cluster that was not chosen for bi-partitioning. This method can be iterated until an optimal quality score or a desired number of clusters is achieved.

---

## III. Higher-Order Recursive Low-Rank + Sparse Structure Learning (HO-RLSL)

### A. Problem Statement

In this paper, we will represent dynamic functional connectivity networks across subjects as a three-way dynamic tensor $\mathcal{M}_t \in \mathbb{R}^{N_1 \times N_2 \times N_3}$. We will assume that this tensor has a low-rank structure $\mathcal{L}_t$ with $\text{rank}(L^{(i)}_t) \ll \min(N_i, \prod_{k=1, k \neq i}^{3} N_k)$ along the connectivity and subject modes corresponding to the modular network structure plus some sparse outlier connections $\mathcal{S}_t$ as:

$$\mathcal{M}_t = \mathcal{L}_t + \mathcal{S}_t \tag{11}$$

Our goal is to separate the low-rank tensor $\mathcal{L}_t$ from its noisy version $\mathcal{M}_t$. This goal leads to the following optimization problem where the Tucker rank of $\mathcal{L}_t$ is minimized while simultaneously minimizing the $\ell_1$-norm of the noise part, $\mathcal{S}_t$:

$$\min_{\mathcal{L}_t, \mathcal{S}_t} \|\mathcal{L}_t\|_* + \lambda \|\mathcal{S}_t\|_1 \quad \text{s.t.} \quad \mathcal{L}_t + \mathcal{S}_t = \mathcal{M}_t \tag{12}$$

where $\|\mathcal{L}_t\|_*$ is the nuclear norm of the low-rank tensor.

Since minimizing the nuclear norm is an NP hard problem, HoRPCA replaces it by its convex surrogate which is the sum of the nuclear norms of the mode-$i$ unfoldings: $\sum_{i=1}^{3} \|L^{(i)}_t\|_*$. One way to solve this optimization problem and obtain the low-rank and sparse parts of $\mathcal{M}_t$ is to use HoRPCA. However, there are two main drawbacks of applying HoRPCA for streaming or time-varying tensor data. First, it is very time consuming to compute HoRPCA in batch mode since all of the high dimensional data needs to be stored and processed. Second, HoRPCA yields different low-rank subspace information at each time point and subspace tracking as desired in this paper requires additional metrics to compare the subspaces across time. To improve the computation efficiency and to better capture the evolving dynamics of the data, we propose to adapt and extend the projection based subspace update approach outlined in REPROCS. Thus, the optimization problem in (12) is solved in two steps. First, instead of determining the low-rank subspace at each time point, we propose to update the subspace across time by minimizing the Tucker norm, or the nuclear norm of each unfolding. As shown previously, the minimizer of the nuclear norm can be obtained through singular value thresholding (SVT). We will use this approach to determine the low-rank subspace. In step 2, we project the observed tensor, $\mathcal{M}_t$, to a subspace orthogonal to the estimated low-rank subspace and transform (11) to a sparse recovery in noise problem to obtain $\mathcal{S}_t$.

### B. Algorithm Description

Suppose that we have a sequence of training tensors defined as $\mathcal{M}_\text{train}$ which do not contain any sparse information and are used for the initial estimate of the subspace in which each mode of $\mathcal{L}_t$ lies in. In the case of dynamic FCNs constructed from task-based EEG, this may correspond to the pre-stimulus activity. $\mathcal{M}_\text{train} \in \mathbb{R}^{N_1 \times N_2 \times N_3 \times t_\text{train}}$ can be considered as a 4-way tensor where the time information constitutes the 4th mode and its full Tucker decomposition is:

$$\mathcal{M}_\text{train} = \mathcal{C} \times_1 P^{(1)}_0 \times_2 P^{(2)}_0 \times_3 P^{(3)}_0 \times_4 P^{(4)}_0 \tag{13}$$

where $P^{(1)}_0, P^{(2)}_0, P^{(3)}_0$ and $P^{(4)}_0$ are the basis matrices along each mode with $P^{(i)}_0 \in \mathbb{R}^{N_i \times N_i}$. Let $\hat{P}^{(i)}_0$s be the truncated version of $P^{(i)}_0$ obtained by keeping the columns with the singular values greater than $\sigma_\text{min}$. $\hat{P}^{(i)}_0 \in \mathbb{R}^{N_i \times r^{(i)}_0}$ where $i \in \{1, 2, 3\}$ give the initial subspace information for $\mathcal{L}_t$ and $r^{(i)}_0$ is the rank of $\hat{P}^{(i)}_0$. The goal is to estimate $\mathcal{L}_t$ and $\mathcal{S}_t$ for each $t > t_\text{train}$ by recursively updating its corresponding basis $P^{(i)}_t$s. The $\mathcal{L}_t$'s are assumed to satisfy a slowly changing low-rank subspace model which will be detailed in Section III-C.

---

**Algorithm 1: Higher-order Recursive Low-Rank + Sparse Structure Learning**

```
Input:  M_t, P̂^(i)_0 s
Output: L̂_t, Ŝ_t, t_j

for t > 0 do
    for i = 1:3 do
        φ^(i)_t = I − P̂^(i)_{t−1} (P̂^(i)_{t−1})ᵀ
    end for
    Y_t = M_t ×₁ φ^(1)_t ×₂ φ^(2)_t ×₃ φ^(3)_t
    Recover Ŝ_t from Y_t by using GTCS-S algorithm
    Estimate L̂_t ← M_t − Ŝ_t
    if mod(t − t_j + 1, α) = 0 then
        for i = 1:3 do
            D^(i) = [L̂^(i)_{t_j+(k−1)α} ··· L̂^(i)_{t_j+kα−1}]
            P̂^(i)_(t) = deleteDirection(D, P̂^(i)_(t−1))
            P̂^(i)_(t) = addDirection(D, P̂^(i)_(t))
        end for
        if P̂^(1)_(t) ≠ P̂^(1)_(t−1) or P̂^(2)_(t) ≠ P̂^(2)_(t−1) or P̂^(3)_(t) ≠ P̂^(3)_(t−1) then
            j ← j + 1, t_j ← t
            P̂^(1)_(j) ← P̂^(1)_(t), P̂^(2)_(j) ← P̂^(2)_(t), P̂^(3)_(j) ← P̂^(3)_(t)
        end if
    else
        P̂^(1)_(t) ← P̂^(1)_(t−1), P̂^(2)_(t) ← P̂^(2)_(t−1), P̂^(3)_(t) ← P̂^(3)_(t−1)
    end if
end for
```

---

Let $\mathcal{P}_t$ be the set of projection matrices which form the basis for the subspaces in which each mode of $\mathcal{L}_t$ lies in $\mathcal{P}_t = \{P^{(1)}_t, P^{(2)}_t, P^{(3)}_t\}$. Assume $\mathcal{P}_t$ has been accurately predicted using past estimates of $\mathcal{L}_t$ such that the projection of the new basis at time $t$ to the orthogonal complement of the past estimates $\left\|(I - \hat{P}^{(i)}_{t-1}(\hat{P}^{(i)}_{t-1})^\top) P^{(i)}_t\right\|_2$ is small. Then $\mathcal{M}_t$ is projected to the space orthogonal to $\hat{P}^{(i)}_{t-1}$s defined through the projection operators $\phi^{(i)}_t = I - \hat{P}^{(i)}_{t-1}(\hat{P}^{(i)}_{t-1})^\top$ to obtain $\mathcal{Y}_t$ as $\mathcal{Y}_t = \mathcal{M}_t \times_1 \phi^{(1)}_t \times_2 \phi^{(2)}_t \times_3 \phi^{(3)}_t$, which can be rewritten as:

$$\mathcal{Y}_t = (\mathcal{L}_t + \mathcal{S}_t) \times_1 \phi^{(1)}_t \times_2 \phi^{(2)}_t \times_3 \phi^{(3)}_t$$

$$\mathcal{Y}_t = \beta_t + \mathcal{S}_t \times_1 \phi^{(1)}_t \times_2 \phi^{(2)}_t \times_3 \phi^{(3)}_t \tag{14}$$

where $\beta_t = \mathcal{L}_t \times_1 \phi^{(1)}_t \times_2 \phi^{(2)}_t \times_3 \phi^{(3)}_t$. Since $\|\phi^{(i)}_t P^{(i)}_t\|_2$ is small, the projection of $\mathcal{L}_t$ to $\phi^{(i)}_t$s will yield small $\|\beta_t\|_F$. Notice that, although the projection matrices $\phi^{(i)}_t$'s are of size $N_i \times N_i$, they have rank $N_i - \text{rank}(\hat{P}^{(i)}_t)$. Therefore, obtaining $\mathcal{S}_t$ from $\mathcal{Y}_t$ can be represented as sparse recovery problem in small noise. Since $\hat{P}^{(i)}_t$'s are dense and restricted isometry constants (RIC) of measurement matrices ($\phi^{(i)}_t$) are small, we can accurately recover $\mathcal{S}_t$ from $\mathcal{Y}_t$ by solving following problem:

$$\hat{\mathcal{S}}_t = \arg\min \|\mathcal{S}_t\|_1, \quad \text{s.t.} \quad \|\mathcal{Y}_t - \mathcal{S}_t \times_1 \phi^{(1)}_t \times_2 \phi^{(2)}_t \times_3 \phi^{(3)}_t\|_F \leq \epsilon \tag{15}$$

To recover $\mathcal{S}_t$ from $\mathcal{Y}_t$, we apply serial recovery procedure for compressed tensors, known as generalized tensor compressive sensing - serial (GTCS-S). This algorithm repeatedly unfolds the compressed tensor along one of the modes and applies $\ell_1$ optimization to recover its columns. Once $\hat{\mathcal{S}}_t$ is recovered, $\mathcal{L}_t$ can be estimated as $\hat{\mathcal{L}}_t = \mathcal{M}_t - \hat{\mathcal{S}}_t$.

---

**Algorithm 2: deleteDirection**

```
Input:  D: data, P: input basis matrix
Output: Q: output basis matrix

λ = (1/w) diag(PᵀD(PᵀD)ᵀ)  where w is the number of columns of D
i = find(λ < σ_min)
Q = [P \ P(i, :)]
```

**Algorithm 3: addDirection**

```
Input:  D: data, P: input basis matrix
Output: Q: output basis matrix

Projection: compute D_proj ← (I − PPᵀ)D
PCA: compute (1/w) D_proj D_proj^ᵀ = UλUᵀ  where w is the number of columns in D
i = find(diag(λ) > σ_min)
Q = [P  U(i, :)]
```

---

### C. Slowly Changing Subspace & Change Points

The following assumptions are made to define slowly changing subspace along each mode of the tensor:

1. Let $t_j$ denote the change points of the low-dimensional subspaces that $L^{(i)}_t$s are in. Note that the subspaces along each mode can vary independently from the others and as such $t_j$s are the collection of all change points across modes. Assume that for $\tau$ large enough, any $\tau$ length subsequence of $L^{(i)}_t$s lies in low-dimensional subspaces, i.e. $\max_t \text{rank}([L^{(i)}_{t-\tau+1} \cdots L^{(i)}_t]) \ll \min(\tau, N_i, \prod_{k=1, k \neq i}^{3} N_k)$.

2. $\mathcal{L}_t$ lies in a low dimensional subspace that changes slowly along each mode $i$ i.e. $\mathcal{L}_t = \mathcal{A}_t \times_1 P^{(1)}_t \times_2 P^{(2)}_t \times_3 P^{(3)}_t$ with $P^{(i)}_t = P^{(i)}_j$ for all $t_j \leq t \leq t_{j+1}$, $j = 1, 2, \ldots, J$ where $J$ is the maximum number of change points. $P^{(i)}_j$ is an $N_i \times r^{(i)}_j$ basis matrix where $r^{(i)}_j \ll \min(N_i, \prod_{k=1, k \neq i}^{3} N_k)$.

3. At the change points, $t_j$, at least one of the $P^{(i)}_j$'s changes as $P^{(i)}_j = [P^{(i)}_{j-1}\ P^{(i)}_{j,\text{add}}]$, $P^{(i)}_j = [P^{(i)}_{j-1} \setminus P^{(i)}_{j,\text{del}}]$ or $P^{(i)}_j = [(P^{(i)}_{j-1} \setminus P^{(i)}_{j,\text{del}}),\ P^{(i)}_{j,\text{add}}]$ where $P^{(i)}_{j,\text{add}}$ is a $N_i \times c^{(i)}_{j,\text{add}}$ basis matrix with $(P^{(i)}_{j,\text{add}})^\top P^{(i)}_{j-1} = 0$, i.e., the new directions added to the projection matrix are orthogonal to the previous directions and $P^{(i)}_{j,\text{del}}$ is a $N_i \times c^{(i)}_{j,\text{del}}$ matrix of deleted basis columns.

4. There exist constants $c^{(i)}_\text{max}$ such that $0 \leq c^{(i)}_{j,\text{add}} \leq c^{(i)}_\text{max} < r^{(i)}_0$. The condition $0 \leq \sum_{i=1}^{j}(c_{i,\text{add}} - c_{i,\text{del}}) \leq c^{(i)}_\text{dif}$ is required to imply $r^{(i)}_t \leq r^{(i)}_0 + c^{(i)}_\text{dif} := r^{(i)}_\text{max}$.

5. The projection of $\mathcal{L}_t$ along the new added directions, $\mathcal{A}_{t,\text{add}} = \mathcal{L}_t \times_1 P^{\prime(1)}_{j,\text{add}} \times_2 P^{\prime(2)}_{j,\text{add}} \times_3 P^{\prime(3)}_{j,\text{add}}$ is initially small, i.e. $\max_{t_j \leq t \leq t_j + \alpha} \|\mathcal{A}_{t,\text{add}}\|_\infty \leq \gamma_\text{add}$ and $\gamma_\text{add} \ll \min(\|\mathcal{L}_t\|_F, \|\mathcal{S}_t\|_F)$, but can increase gradually.

In order to enable a more efficient online implementation, the low-rank subspaces $P^{(i)}_t$s are estimated and updated every $\alpha$ samples, where $\alpha$ is selected empirically. Similar to the projection PCA (p-PCA) procedure, mode-$i$ unfoldings $\hat{L}^{(i)}_t$'s of the last $\alpha$ $\hat{\mathcal{L}}_t$'s are concatenated as $D^{(i)} = [\hat{L}^{(i)}_{t_j+(k-1)\alpha} \cdots \hat{L}^{(i)}_{t_j+k\alpha-1}]$ with $k \in \{1, 2, \ldots, K\}$ where $K$ is the maximum number of length $\alpha$ windows, and $D^{(i)}$s are projected onto subspaces which are orthogonal to $\hat{P}^{(i)}_{(j-1)}$s as follows: $D^{(i)}_\text{proj} = (I - \hat{P}^{(i)}_{(j-1)}(\hat{P}^{(i)}_{(j-1)})^\top) D^{(i)}$. Then PCA is applied to find the subspace which spans $D^{(i)}_\text{proj}$. Let $P^{(i)}_{j,\text{add}}$ be the truncated basis that spans this subspace obtained by keeping the eigenvectors with eigenvalues greater than $\sigma_\text{min}$. $P^{(i)}_{j,\text{add}}$ and previous subspace estimate $\hat{P}^{(i)}_{(j-1)}$ together yield the new subspace estimate as: $\hat{P}^{(i)}_{(j)} = [\hat{P}^{(i)}_{(j-1)}\ P^{(i)}_{j,\text{add}}]$. During the update step, some of the existing directions can also be deleted from the projection matrix by finding the ones with eigenvalues lower than $\sigma_\text{min}$ (see Algorithms 2 and 3). If there are any added or deleted directions, it means that there is a change point. It is also important to note that, the tensor subspace estimation implemented in this paper finds subspaces along each mode individually without taking other modes into account. Thus, the proposed method is not optimized like higher-order orthogonal iteration (HOOI) but offers a computationally efficient way of estimating subspaces along each mode.

### D. Computational Complexity

In this section, we offer a comparison of the computational complexity of the proposed approach with respect to REPROCS applied to our data in vectorized form. Let the 3-way tensor be of size $N \times N \times N$. For the time points which do not require subspace update, computational complexity of the proposed approach is equivalent to the complexity of $\ell_1$ regularization $O(N^3)$ multiplied by the total number of fibers to recover for each mode to obtain the sparse component and is equal to $3N^2 O(N^3)$. However, if we use REPROCS after vectorizing the data, complexity for the same operations become $O((N^3)^3) = O(N^9)$. For the time points which require basis update, there is an additional cost of covariance matrix computation and eigenvalue decomposition. For our approach, covariance matrix computations for the three modes have a complexity of $3O((\alpha N) \times N^4) = 3O(\alpha N^5)$ operations whereas eigenvalue decompositions cost $3O(N^3)$. However, REPROCS requires $O(\alpha(N^3)^2) = O(\alpha N^6)$ operations for covariance matrix computation and $O((N^3)^3) = O(N^9)$ operations for eigenvalue decomposition.

---

## IV. Results

### A. Simulated Networks

The proposed framework is first applied to three simulated dynamic tensors $\mathcal{X}_t \in \mathbb{R}^{64 \times 64 \times 60}$ for $t \in \{1, 2, \ldots, 80\}$. For each network type, 20 simulations of the tensors are generated where each frontal slice $\mathcal{X}_t(:,:,i)$ corresponds to a weighted and undirected network. In our experiments, to generate the networks, we used three well-known network models known as the modular small-world network, hierarchical modular small-world network and overlapped modular networks. Distinct regions in the brain which are strongly connected within themselves are specialized for different processes in the brain and, this phenomenon is known as functional segregation. Presence of these specialized neuronal groups appear as different modules in brain networks. Moreover, some of the nodes in a module may have more specialized function which yields hierarchical structure in a network while some of the nodes belong to multiple clusters resulting in overlapping modules.

For the experiments including modular small world network model, initially, the networks contain 2 equal size modules. After $t = 20$, both of the modules are slowly divided into two smaller modules of size 16 nodes each. After $t = 60$, the network structure evolves back to the initial structure. For the second set of experiments with the hierarchical modular small world network model, the networks contain 2 equal size modules at the beginning. After $t = 20$, both of the modules are slowly divided into two smaller modules of size 16 nodes each while establishing hierarchical structure. After $t = 60$, the network structure evolves back to the initial structure. For the third set of experiments with overlapping modules, the networks contain 2 equal size non-overlapping modules at the beginning. After $t = 20$, 25% of the nodes start to belong to both modules. After $t = 60$, network structure evolves back to the initial non-overlapping structure.

For all of the experiments, intra-cluster edge values were selected from $\mathcal{N}(0.6, 0.1)$ and truncated to the interval $[0, 1]$ while the inter-cluster edge values were selected from $\mathcal{N}(0.1, 0.1)$. Moreover, these networks were corrupted by a sparse noise matrix $E_t$ whose sparsity varies from 10% to 40% and $e_{i,j} \sim \text{beta}(4, 2)$. Proposed algorithm is applied with $\alpha = 5$ and $\sigma_\text{min}$ is determined as 10% of highest singular value obtained from the initial subspace estimate along that mode using the first 5 time points. The proposed algorithm is compared to an implementation without the sparse recovery step similar to performing standard HoSVD at each time point. Mean squared error which quantifies the error between estimated and original low-rank components for all of the network models are computed for both algorithms as:

$$\text{MSE} = \frac{1}{T_\text{end} - T_\text{start} + 1} \sum_{t=T_\text{start}}^{T_\text{end}} \frac{\|\mathcal{L}_t - \hat{\mathcal{L}}_t\|^2_F}{\prod_{i=1}^{3} N_i}$$

where $T_\text{start}$ and $T_\text{end}$ are the start and end points of the detected time interval and $N_i$ is the size of the tensor along $i$th mode.

**Table I. Average MSE for Modular Network Structure under Varying Noise Sparsity Levels**

| Noise Level | Method  | Interval-1 | Interval-2 | Interval-3 |
|-------------|---------|------------|------------|------------|
| 10%         | Ho-RLSL | 0.0046     | 0.0094     | 0.0045     |
|             | HoSVD   | 0.0097     | 0.0144     | 0.0095     |
| 20%         | Ho-RLSL | 0.0085     | 0.0167     | 0.0084     |
|             | HoSVD   | 0.0233     | 0.0302     | 0.0229     |
| 30%         | Ho-RLSL | 0.0136     | 0.0260     | 0.0137     |
|             | HoSVD   | 0.0409     | 0.0496     | 0.0405     |
| 40%         | Ho-RLSL | 0.0215     | 0.0387     | 0.0216     |
|             | HoSVD   | 0.0613     | 0.0713     | 0.0607     |

**Table II. Average MSE for Hierarchical Modular Network Structure under Varying Noise Sparsity Levels**

| Noise Level | Method  | Interval-1 | Interval-2 | Interval-3 |
|-------------|---------|------------|------------|------------|
| 10%         | Ho-RLSL | 0.0149     | 0.0265     | 0.0137     |
|             | HoSVD   | 0.0202     | 0.0319     | 0.0189     |
| 20%         | Ho-RLSL | 0.0184     | 0.0333     | 0.0171     |
|             | HoSVD   | 0.0335     | 0.0465     | 0.0322     |
| 30%         | Ho-RLSL | 0.0240     | 0.0377     | 0.0225     |
|             | HoSVD   | 0.0511     | 0.0663     | 0.0497     |
| 40%         | Ho-RLSL | 0.0346     | 0.0521     | 0.0326     |
|             | HoSVD   | 0.0712     | 0.0875     | 0.0699     |

**Table III. Average MSE for Network Structure with Overlapping Modules under Varying Noise Sparsity Levels**

| Noise Level | Method  | Interval-1 | Interval-2 | Interval-3 |
|-------------|---------|------------|------------|------------|
| 10%         | Ho-RLSL | 0.0043     | 0.0088     | 0.0045     |
|             | HoSVD   | 0.0091     | 0.0107     | 0.0094     |
| 20%         | Ho-RLSL | 0.0080     | 0.0195     | 0.0083     |
|             | HoSVD   | 0.0225     | 0.0250     | 0.0228     |
| 30%         | Ho-RLSL | 0.0131     | 0.0336     | 0.0135     |
|             | HoSVD   | 0.0399     | 0.0435     | 0.0404     |
| 40%         | Ho-RLSL | 0.0209     | 0.0509     | 0.0215     |
|             | HoSVD   | 0.0602     | 0.0640     | 0.0606     |

Tables I–III show that Ho-RLSL is more robust than HoSVD for sparse outliers with smaller MSE values. As the sparsity level of the noise increases, the difference in performance between the two algorithms also increases. Complexity of the network structure, i.e., modular vs. hierarchically modular, also affects the accuracy of the algorithms and increased structural complexity results in increased error.

### B. Effect of Network Size on Computation Time and Performance

The proposed framework is applied to two simulated dynamic tensors $\mathcal{X}_t \in \mathbb{R}^{64 \times 64 \times 60}$ and $\mathcal{X}_t \in \mathbb{R}^{128 \times 128 \times 60}$ for $t \in \{1, 2, \ldots, 80\}$ to see the effect of network size on the computation time and performance of the algorithm. 10 simulations of the tensors are generated where each frontal slice $\mathcal{X}_t(:,:,i)$ corresponds to a weighted and undirected network and the third mode corresponds to the number of subjects. In these experiments, to generate the networks, we used modular small-world networks. For the experiments, initially, the networks contain 2 equal size modules. After $t = 20$, both of the modules are slowly divided into two smaller modules of equal size. After $t = 60$, the network structure evolves back to the initial structure.

For the experiments, intra-cluster edge values were selected from $\mathcal{N}(0.6, 0.1)$ and truncated to the interval $[0, 1]$ while the inter-cluster edge values were selected from $\mathcal{N}(0.1, 0.1)$. Moreover, these networks were corrupted by a sparse noise matrix $E_t$ whose sparsity is 10% and $e_{i,j} \sim \text{beta}(4, 2)$. Proposed algorithm is applied with $\alpha = 5$ and $\sigma_\text{min}$ is determined as 10% of highest singular value obtained from the initial subspace estimate along that mode using the first 5 time points.

All of the simulations were run on a computer with Intel(R) Core(TM) i5-2500T CPU and 4.00 GB memory.

**Table IV. Average Computation Time for Ho-RLSL for Dynamic Tensors Containing 64×64 and 128×128 Networks**

| Tensor Size      | Time (sec)           | Interval-1 | Interval-2 | Interval-3 |
|------------------|----------------------|------------|------------|------------|
| 64 × 64 × 60    | $2.3916 \times 10^4$ | 0.0046     | 0.0090     | 0.0045     |
| 128 × 128 × 60  | $6.3397 \times 10^4$ | 0.0045     | 0.0065     | 0.0046     |

Table IV shows that doubling the number of nodes in the network increases the computation time almost three times. Moreover, increased network size yields better low-rank estimation for the complex network structures. As seen in Table IV, MSE computed for the second time interval where the networks contain more modules significantly decreases with the increased network size. This is due to the fact that with more modules in the network the number of nodes in a module decreases making the subspace estimation more challenging. When the number of nodes in the network increases, the subspace estimation becomes more accurate.

### C. EEG Data

The proposed tensor tracking approach is applied to a set of connectivity graphs constructed from EEG data containing the error-related negativity (ERN) and correct-related negativity (CRN). The ERN is a brain potential response that occurs following performance errors in a speeded reaction time task usually 25–75 ms after the response. Previous work indicates that there is increased coordination between the lateral prefrontal cortex (lPFC) and medial prefrontal cortex (mPFC) within the theta frequency band (4–8 Hz) and ERN time window.

EEG data from 63-channels was collected in accordance with the 10/20 system on a Neuroscan Synamps2 system sampled at 128 Hz from 91 subjects. The task was a common speeded-response letter (H/S) flanker, where error and correct response-locked trials from each subject were utilized. A random subset of correct trials was selected, to equate the number of error relative to correct trials for each participant. The EEG data are pre-processed by the spherical spline current source density (CSD) waveforms to sharpen event-related potential (ERP) scalp topographies and reduce volume conduction. The CSD has fewer assumptions than many inverse transforms, attenuates volume conduction, and represents independent sources near the cortical surface. For each subject and response type, the pairwise phase locking value in the theta frequency band was computed as described in (8). We constructed 3-way tensors at each time point $\mathcal{X}_t \in \mathbb{R}^{63 \times 63 \times 91}$ for both ERN and CRN data separately where the first and second mode represent the adjacency matrix of the connectivity graphs while the third mode corresponds to the subjects for $t \in \{1, 2, \ldots, 256\}$.

The connectivity networks corresponding to the first 10 time points $t \in \{1, 2, \ldots, 10\}$ and all subjects was used in the training step to obtain initial subspace information of the low-rank component $\mathcal{L}_t$. This training data is used to obtain an initial subspace estimate and 10 time points approximately correspond to 78 ms of data. Then the proposed approach was applied to the remaining time points with $\alpha = 8$ and $\sigma_\text{min} = 0.11$. Since the connectivity networks constructed by the phase synchrony measure are symmetric, the basis matrices $P^{(1)}_t$ and $P^{(2)}_t$ corresponding to the first two modes are identical to each other as $P^{(1)}_t = P^{(2)}_t$ for each time point $t$.

Change points corresponding to the connectivity mode $L^{(1)}_t$ show that for ERN networks there is an interval $(-109, 141)$ ms corresponding to the response time and the ERN response. There is also a longer interval $(141, 766)$ ms corresponding to the Pe, the error-related positivity which usually occurs 200–500 ms after making an incorrect response, following the error negativity (ERN). Similar time intervals are obtained for CRN, with the biggest difference being a longer time interval around the response time as the physiological response for CRN is not as pronounced as the one for ERN.

In order to better interpret the network structure corresponding to different time intervals, FCCA was applied to the sets of $63 \times 63$ networks obtained from low-rank tensors $\hat{\mathcal{L}}_t \in \mathbb{R}^{63 \times 63 \times 91}$ within each time interval (pre-ERN, ERN, post-ERN, pre-CRN, CRN, post-CRN) where (# of input adjacency matrices) = (# of subjects) × (# time points in the interval). The network structure in pre-ERN interval is similar to the network structure of the post-ERN interval, while the identified modules are more segregated in the ERN interval relative to the pre and post-ERN. This is in line with previous results indicating that separable clusters are apparent relative to left and right motor areas, and left and right lateral-PFC regions during ERN. For the CRN clusters, it was observed that segregation of the lateral and central areas is quite limited with one large fronto-central cluster present during all CRN intervals. The exception are small frontal- and central clusters during the CRN. This is consistent with the idea that medial frontal regions are activated during correct trials, but to a smaller extent than during errors.

**Table V. Detected ERN and CRN Intervals by Ho-RLSL and HoSVD**

| Interval  | Ho-RLSL                   | HoSVD                      |
|-----------|---------------------------|----------------------------|
| Pre-ERN   | −0.484 to −0.109 ms       | −0.484 to −0.047 ms        |
| ERN       | −0.109 to 0.141 ms        | −0.047 to 0.141 ms         |
| Post-ERN  | 0.141 to 0.766 ms         | 0.141 to 0.766 ms          |
| Pre-CRN   | −0.422 to −0.047 ms       | −0.734 to −0.109 ms        |
| CRN       | −0.047 to 0.391 ms        | −0.109 to 0.141 ms         |
| Post-CRN  | 0.391 to 0.641 ms         | 0.141 to 0.391 ms          |

To show the denoising performance of the proposed method compared to HoSVD, HoSVD was applied to the EEG networks with the same $\alpha$ and $\sigma_\text{min}$ values used in Ho-RLSL. First, HoSVD detected pre-ERN, ERN and post-ERN intervals very similar to the ones obtained from Ho-RLSL (see Table V). The change point at the end of the pre-ERN interval was shifted by $\alpha$ time points. Both HoSVD and Ho-RLSL yield the same network structure for the ERN interval, while HoSVD yields more noisy networks for pre- and post-ERN intervals. When the physiological response is strong such as during the ERN, the network is less noisy yielding the exact cluster structure both for Ho-RLSL and HoSVD. However, when the networks are noisier such as for the pre-ERN interval, our algorithm provides a cleaner low-rank approximation which yields more distinct cluster structures. For CRN networks, Ho-RLSL and HoSVD detected very different pre-CRN, CRN and post-CRN intervals, and in particular HoSVD detects a very long pre-CRN time interval and very short CRN and post-CRN intervals. The CRN and post-CRN intervals detected by Ho-RLSL align better with well-known ERPs such as P300.

---

## V. Conclusion

In this paper, we introduced a new recursive low-rank + sparse structure learning algorithm for tensor type data to track dynamic modular structure of functional connectivity networks constructed from EEG recordings across multiple subjects. To this aim, a recent subspace tracking approach, REPROCS, was adapted and extended to tensor type data. This extension offers several novelties with respect to the original algorithm. In original REPROCS, the measurements at each time point are vectors and the algorithm uses the fact that each measurement vector is coming from a low-rank subspace. However, in our case, low-rank corresponds to a low Tucker rank, i.e. the matricized version of the tensor along each mode is a low-rank matrix implying it can be reconstructed through the outer product of a small number of eigenvectors. This is particularly suitable for dealing with data that has a modular structure such as FCNs. Using REPROCS to analyze this type of dataset would require vectorization which breaks the intrinsic low-rank structure of each measurement along with increased computational complexity. The proposed approach yields robust estimation of the low-rank component of a dynamic 3-way tensor at each time point and provides a recursive way to update the subspace information. This approach identifies the time points where the low-rank subspaces change and recursively updates these subspaces to improve the low-rank approximation to data. The proposed approach is first applied to a set of simulated networks for performance evaluation, and is then used to separate low-rank and sparse parts of dFCNs across multiple subjects. Low-rank component of dFCNs obtained for the detected time intervals are summarized by using a recently introduced multiple graph clustering approach, FCCA. The low-rank subspace approximation to each time interval provides a denoised version of the original network, thus improving the quality of the clustering results.

The results indicate a clear change in the community structure before and after the physiological response. Moreover, the detected community structures are in line with previous hypothesis regarding the networks involved in cognitive control.

One main concern about the proposed algorithm is the selection of parameters $\alpha$ and $\sigma_\text{min}$. Both $\alpha$ and $\sigma_\text{min}$ are data-dependent parameters and their selection requires some a priori information. First, $\alpha$ should be smaller than the time interval between two consecutive change points in order to identify each one of them. For example, in our simulations change points appear at $t = 40$ and $t = 60$ and there are 20 time points between consecutive change points. We selected $\alpha$ as 5 in our simulations and we identified the change points correctly. When we apply our algorithm with $\alpha = 10$ we can still identify the change points. However, if we select $\alpha = 30$, we identify the first change point at $t = 60$ which corresponds to the beginning of the third interval. Therefore, the subspace estimate is based on the information from the second time interval and cannot do a good job of representing the third time interval and the algorithm fails. Moreover, if we keep $\alpha$ very small, then the algorithm will be more susceptible to instantaneous noise and we will have a lot of incorrect change points or false positives. For the real dataset, since we know something about the dynamics of EEG, we selected $\alpha$ to be at least as long as the duration of ERNs (50–75 ms) with $\alpha = 8$. Selection of $\sigma_\text{min}$ is also dependent on the datasets. For example, in our simulations, we selected $\sigma_\text{min}$ as 10% of the maximum eigenvalue. If we select it too small, the algorithm includes many basis vectors which correspond to noise and both the low-rank assumption and the algorithm fail. If we select $\sigma_\text{min}$ too high, then the algorithm does not update the low-rank subspace for the slow changes and the algorithm again fails. Selecting $\sigma_\text{min}$ for real dataset is more difficult, because gaps between eigenvalues may not be very clear. For our EEG datasets, the first eigenvalue was much larger than the others and using a threshold around 10% of the maximum eigenvalue did not work. In this case, we chose $\sigma_\text{min} = 0.11$ empirically.

Future work will consider extending the proposed approach to higher order tensors by including various experimental conditions, different frequency bands and multiple modalities. Future work will also consider extensions of linear low-rank subspace models to unions of subspaces or manifolds. Modifications to the Ho-RLSL algorithm such as predicting support of the sparse component or using partial subspace knowledge for the low-rank component will further improve the performance of Ho-RLSL.

---

## Appendix

### A. $\beta_t$ Is Small

In this section, we will prove that $\beta_t$ is small and that (14) can be treated as a sparse recovery in noise problem. Define the subspace estimation error for $i$th mode as:

$$SE(P^{(i)}, \hat{P}^{(i)}) := \|(I - \hat{P}^{(i)}\hat{P}^{(i)\top})P^{(i)}\|_F = \epsilon_i \tag{16}$$

where $P^{(i)}$ and $\hat{P}^{(i)}$ are true and estimated basis matrices of the $i$th mode, respectively. $P^{(i)}_j = [P^{(i)}_{j-1}\ P^{(i)}_{j,\text{new}}]$ where $P^{(i)}_{j,\text{new}}$ is a $n_i \times c^{(i)}_{j,\text{new}}$ basis matrix with $(P^{(i)}_{j,\text{new}})^\top P^{(i)}_{j-1} = 0$. Since the low-rank tensor at time $t$, $\mathcal{L}_t$, has components $A_m$s with $m \in \{1, 2, \ldots, 8\}$ which are the projections of $\mathcal{L}_t$ in both the previous subspace $P^{(i)}_{j-1}$s and $P^{(i)}_{j,\text{new}}$s. Therefore, $\mathcal{L}_t$ can be written as the sum of these components:

$$\mathcal{L}_t = A_1 \times_1 P^{(1)}_{j-1} \times_2 P^{(2)}_{j-1} \times_3 P^{(3)}_{j-1}$$
$$+ A_2 \times_1 P^{(1)}_{j,\text{new}} \times_2 P^{(2)}_{j-1} \times_3 P^{(3)}_{j-1}$$
$$+ A_3 \times_1 P^{(1)}_{j-1} \times_2 P^{(2)}_{j,\text{new}} \times_3 P^{(3)}_{j-1}$$
$$+ A_4 \times_1 P^{(1)}_{j-1} \times_2 P^{(2)}_{j-1} \times_3 P^{(3)}_{j,\text{new}}$$
$$+ A_5 \times_1 P^{(1)}_{j,\text{new}} \times_2 P^{(2)}_{j,\text{new}} \times_3 P^{(3)}_{j-1}$$
$$+ A_6 \times_1 P^{(1)}_{j,\text{new}} \times_2 P^{(2)}_{j-1} \times_3 P^{(3)}_{j,\text{new}}$$
$$+ A_7 \times_1 P^{(1)}_{j-1} \times_2 P^{(2)}_{j,\text{new}} \times_3 P^{(3)}_{j,\text{new}}$$
$$+ A_8 \times_1 P^{(1)}_{j,\text{new}} \times_2 P^{(2)}_{j,\text{new}} \times_3 P^{(3)}_{j,\text{new}} \tag{17}$$

**Assumptions:**

1. Assume that subspace estimation error is $\epsilon_i = \|(I - \hat{P}^{(i)}\hat{P}^{(i)\top})P^{(i)}\|_F \leq r^{(i)}_0 \zeta$ for $\zeta \ll 1$.
2. Let $l^{(k)}_i$ be the $i$th column of $L_{t,(k)}$ and assume that $\|l^{(k)}_i\|_F \leq \gamma_{*,k}$, $\gamma_{*,k} \leq \frac{1}{\sqrt{\zeta r^{(k)}_J}}$ and $\gamma_{\text{new},k} \ll \gamma_{*,k}$.
3. Define $\gamma_* = \min_k(\gamma_{*,k} \prod_{i=1, i\neq k}^{3} N_i)$ and assume that $\|\mathcal{L}_t\|_F \leq \gamma_*$.
4. Define $\gamma_\text{new} = \min_k(\gamma_{\text{new},k} \prod_{i=1, i\neq k}^{3} N_i)$ and assume that $\gamma_\text{new} \ll \gamma_*$.

$\beta_t$ is defined as $\beta_t = \mathcal{L}_t \times_1 \phi^{(1)}_t \times_2 \phi^{(2)}_t \times_3 \phi^{(3)}_t$ where $\phi^{(i)}_t = I - \hat{P}^{(i)}_{t-1}(\hat{P}^{(i)}_{t-1})^\top$ and its norm is:

$$\|\beta_t\|_F \leq \epsilon_1\epsilon_2\epsilon_3 \|A_{t,*}\|_F + \epsilon_2\epsilon_3\|A_2\|_F + \epsilon_1\epsilon_3\|A_3\|_F + \epsilon_1\epsilon_2\|A_4\|_F$$
$$+ \epsilon_3\|A_5\|_F + \epsilon_2\|A_6\|_F + \epsilon_1\|A_7\|_F + \|A_{t,\text{new}}\|_F \tag{19}$$

Since $\zeta$ is small and $\bar{\gamma}_*$ is large, the last term is dominant in the upper bound. Thus, $\beta_t$ can be considered as a noise by the slow subspace change assumption $\|\gamma_\text{new}\|_F \ll \|\mathcal{S}_t\|_F$.

### B. Upper Bound for Frobenius Norm of Cross Terms

Upper bound for the norm of the cross terms is derived as follows:

$$\|A_{t,2}\|_F = \|\mathcal{L}_t \times_1 P^{\prime(1)}_{j,\text{new}} \times_2 P^{\prime(2)}_{j-1} \times_3 P^{\prime(3)}_{j-1}\|_F$$

$$\leq \|P^{\prime(1)}_{j,\text{new}} L_{t,(1)}\|_F \cdot \|P^{(3)}_{j-1} \otimes P^{(2)}_{j-1}\|_F$$

$$= \gamma_* \sqrt{r^{(1)}_{j,\text{new}} r^{(2)}_{j-1} r^{(3)}_{j-1}} \tag{26}$$

---

*Manuscript received January 12, 2017; accepted January 23, 2016. Date of publication February 13, 2017; date of current version November 3, 2017. This work was supported in part by National Science Foundation under Grant CCF-1422262 and Grant CCF-1218377.*

*A. Ozdemir and S. Aviyente are with the Department of Electrical and Computer Engineering, Michigan State University, East Lansing, MI, 48824 USA. E. M. Bernat is with the Department of Psychology, University of Maryland, College Park, MD, 20742 USA.*
