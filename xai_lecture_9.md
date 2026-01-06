# XAI Lecture 9: Prototypes and Counterfactuals


## Prototypes

### Introduction

- Prototypes are examples that can be used to summarise a large number of cases
- They essentially explain the dataset by collapsing the relevant data points to a manageable number, thus reducing complexity dramatically
- Prototypes are typically considered **global** explanations for the dataset, even if each prototype "explains" only a local part of the dataset

### Finding prototypes

- One way to find prototypes is to use a clustering algorithm and then pick an element per cluster
- For instance, one may use k-means to obtain k clusters
- Prototypes can be of two kinds
	- strictly speaking they are elements of the original dataset
	- more broadly, they can be synthetic examples not in the real data
- Example ways to obtain a prototype strictly from the data are to choose:
	- a cluster element at random,
	- the cluster element closest to the centre of the cluster,
	- the cluster element with the lowest average intra-cluster distance
- Some clustering methods automatically yield a representative element from the cluster
	- an example is k-medoids (same idea as k-means, but without computing the mean and instead trying out each data point as cluster centre)
- If the space used for the analysis can be mapped back to the original data space, then also points that do not correspond to instances can be used as prototypes
	- an example is to just use the cluster centres

- For classification problems, one can divide the analysis per class
	- This is instead or on top of clustering

### MMD-critic

Use Kernel Density Estimation (KDE) to estimate the data density,
which means centring a small probability distribution around each data point,

$$
p_{\\{\vec{x}^{(i)}\\}}(\vec{x}) = \frac{1}{N} \sum_{i=1}^N p_{\vec{x}^{(i)}}(\vec{x}).
$$

For example this could be a Gaussian density

$$
p_{\vec{x}_0}(\vec{x}) = \mathcal{N}(\vec{x}_0, \sigma)(\vec{x}) = \frac{1}{\sqrt{2}\sigma} e^{-\frac{|\vec{x} - \vec{x}_0|^2}{2\sigma^2}},
$$

but also a distribution with fat tails, or a sharp one for that matter

$$
k(\vec{x}, \vec{x}_0) = \theta(\sigma - |\vec{x} - \vec{x}_0|).
$$

The most important parameter of KDE is the bandwidth $\sigma$.


The idea of Maximum Mean Discrepancy is to try to minimize the distance
between the KDE done with prototypes and the KDE with the data.

$$
\text{MMD}^2 = \frac{1}{m^2} \sum_{i,j=1}^m k(\vec{z}_i, \vec{z}_j) - \frac{2}{mn} \sum_{i=1}^m \sum_{j=1}^n k(\vec{z}_i, \vec{x}_j) + \frac{1}{n^2} \sum_{i,j=1}^n k(\vec{x}_i, \vec{x}_j)
$$

$$
\text{MMD}^2 = \mathbb{E}_\mathcal{X}[p_\mathcal{X}]
- \mathbb{E}_\mathcal{Z}[p_\mathcal{X}]
+ \mathbb{E}_\mathcal{Z}[p_\mathcal{Z}]
- \mathbb{E}_\mathcal{X}[p_\mathcal{Z}]
$$

The algorithm to find the prototypes optimises the MMD objective greedily.

- While the number of prototypes is below $k$
	- Check how much the criterion is improved for all points in the dataset.
	- Add the point which leads to the best improvement to the prototypes.

### Criticisms

- Criticisms are data points that are not represented well by prototypes

$$\mathrm{witness}(\mathbf{x})=\frac{1}{n}\sum_{i=1}^{n}k(\mathbf{x}, \mathbf{x}^{(i)})-\frac{1}{m}\sum_{j=1}^{m}k(\mathbf{x}, \mathbf{z}^{(j)})$$

- Criticisms should be **diverse**

### Prototype networks

- Prototype networks were designed for **few-shot learning**, i.e., the task of supervised learning with a very limited number of examples (usually 1-30).
- The idea is that a neural network maps inputs to a **latent space** and a prototype in the latent space for each class is used for nearest neighbour classification.
- The **prototype coordinates** are **learned** using gradient descent, possibly jointly to the network.

- In general, there is no straightforward way to interpret the prototype, as it lives in an abstract space.
- If the encoder to the latent space is invertible with a **decoder**, e.g., because it originates from an autoencoder, then the prototype can be transformed back to the input space and interpreted.


### Discussion

- Finding prototypes is connected to the task that is known as dataset pruning, instance selection, dataset distillation, ...
	- Other methods such as condensed nearest neighbours or edited nearest neighbours can be used.
- Data-centric exploration and debugging method
- Weaknesses
	- metric space where prototypes are defined
	- prototypes and criticisms are arbitrarily separated
	- accuracy/faithfulness loss


## Counterfactual examples

### Introduction

- One of the major features of intelligence is the ability to simulate the future, answering "what happens as a consequence of this action"
- It allows some animals, including some mammals and birds, to explore the consequences of actions to choose the one that leads to the best outcome
- Counterfactual reasoning is more subtle, it's about "what would have happened if the action had been different"
- It's less immediate than "which action is the best for this situation, which is similar to a past one"
- It's a faculty very difficult to disentangle from pure reinforcement learning without language
- Some scientists claim that it is a uniquely human skill

- In many situations humans want to know what would have been the outcome under different conditions
- In the context of XAI, it is useful to consider this **with respect to the models**, i.e., what would have been the output had the input to the model been different
- Importantly, this does not imply that in the real world the outcome would have changed in the same way as the model prediction

- There is a subtle difference between **contrastive** and **counterfactual** examples
- Contrastive = "here's an example where the output is different"
- Counterfactual = "if the input is changed this way, the output would be like that"
- Counterfactual examples are connected to causality since they look at an intervention that may change the outcome, contrastive ones to correlations since they look at other examples with their outcome (especially similar ones with different outcome)

```diff
! Definition
```
Given a model $\hat{f}$ that outputs
$y=\hat{f}(\vec{x})$ for an instance $\vec{x}$, a counterfactual explanation consists of an instance $\vec{x}'$
such that the output for $\hat{f}$ on $\vec{x}'$ is different from $y$, i.e., $y'=\hat{f}(\vec{x}')\ne\hat{f}(\vec{x})$,
and such that the difference between $\vec{x}$ and $\vec{x}'$ is minimal.


### Properties

Counterfactuals should have the following properties.

1. **Validity**: change the output, not give the same outcome.
2. **Similarity** to the query (a.k.a. proximity): preserve the input, not change features a lot.
3. **Diversity** among themselves: differ from each other, not be all equal.
4. **Plausibility** (a.k.a. feasibility): have high likelihood, not be unrealistic.

Sometimes additional attributes are required.

- **Actionability**: enable changing something to obtain different results.
- **Minimality** (a.k.a. sparsity): be the closest examples that allow for the different output.
- **Discriminative power**: enable humans to reproduce the same model outputs.

### Algorithms

- Counterfactuals can be searched by brute force, e.g., using a grid.
- Brute-force search is expensive.
- Optimization is better, and can be achieved in many ways.
- Gradient-descent can be used for differentiable objectives.
- Constraints can be solved.
- Heuristics can also be used.

#### Wachter et al, 2017

Loss:

$$\mathcal{L} = \lambda[\hat{f}(\vec{x}') - y']^2 + d(\vec{x}, \vec{x}')$$

The distance is taken by scaling the $L_1$ distance for each feature with the Median Absolute Deviation for that feature

$$d(\vec{x}, \vec{x}') = \sum_{j=1}^D\frac{|x_j-x_j'|}{\text{MAD}_j}.$$

Instead of choosing $\lambda$, impose the constraint

$$|\hat{f}(\vec{x}') - y'| < \epsilon,$$

and increase $\lambda$ if the constraint is not satisfied at the end of the optimization.

> Note that one could also use a loss term 
> $\theta(\epsilon - |\hat{f}(\vec{x}') - y'|)$
> with a large weight.

Since gradients may not be available, optimization can be done with symplex-based methods such as Nelder--Mead.

Wachter's method only takes validity and similarity into account.

#### Multi-objective

In general the issue is that the criteria are too many for optimisation, and need to be combined with arbitrary weights.
The method that is optimal for one criterion may not be optimal for another, and there are many trade-offs.

One interesting idea is, instead of combining the losses, to keep them all separate, and use a genetic algorithm that determines the Pareto front at each step, and then mutates / recombines from there.

## Resources

For counterfactual explanations, consult

- Guidotti, [Counterfactual explanations and how to find them: literature review and benchmarking](https://doi.org/10.1007/s10618-022-00831-6), 2021

<!--## Exercises

- Two pieces of code that compute clustering, which one generates prototypes-->