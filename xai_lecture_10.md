# XAI Lecture 10: Influence and Concepts


## Influence

### Motivation

- **Feature** methods explain which input features matter.
- **Example** methods explain outputs for certain data points.

**Data attribution** explains **which training examples influenced** an output.

> How would the model output change if a training example were removed or different?

Training points that significantly change a model are called **influential**.
These are **different from prototypes**.
Indeed, when very few data points can be used for training, it's usually best to have typical examples,
but when a lot of data points are available, it's the edge cases that determine the boundaries of behaviour.

This is particularly useful for debugging the training process and data.
Specifically the following may be discovered.

- **Mislabeled** data points that complicate decision boundaries (when this is on purpose it's called *data poisoning*).
- Regions where data is **scarce**, either because they are out of scope or because of underrepresentation (*bias*).
- Issues in **optimisation**, when the model learns from limited or unexpected data subsets.

Given a training dataset $\mathcal{D}$ with samples $\{1, \dots, N\}$,
the **influence** of training sample $j$ is the change it produces in model parameters
or in either prediction or loss function for one or more evaluation samples.

### Deletion diagnostics

Deletion diagnostics measure influence by **actually retraining** the model without a data point.

Consider a model $f_\theta$ with parameters $\theta$,
where the learned parameters are $\hat\theta$ when the model is trained on $\mathcal{D}$
and $\hat\theta_{-j}$ when it is trained on $\mathcal{D}\setminus\{j\}$.
When the training objective is expressed in terms of the loss function $\mathcal{L}$ we have

$$
\hat\theta = \arg\min_\theta \mathcal{L}(\{z^{(i)}\}_{i\in\mathcal{D}},\theta),
\qquad
\hat\theta_{-j} = \arg\min_\theta \mathcal{L}(\{z^{(i)}\}_{i\in\mathcal{D}\setminus\{j\}},\theta).
$$

Here data points are denoted with $z^{i}$ which comprises the features $\vec{x}^{(i)}$ and,
for supervised learning, also the labels $y^{(i)}$.
Traditional influence estimates were, unsurprisingly, designed for linear regression.
A well-known one is difference in beta, in short **DFBETA**,
which is just the difference in the parameter values when removing the $i$-th training instance

$$
\text{DFBETA}_j = \hat\theta - \hat\theta_{-j}.
$$

The name comes from the coefficients in linear regression typically denoted with $\beta$ instead of $\theta$.

Another established influence quantification for linear regression models is **Cook's distance** $D$,
which measures the prediction change

$$
D_j = \frac{1}{p\cdot \mathrm{MSE}}\sum_{i\in\mathcal{D}}
[f_{\hat\theta}(\vec{x}^{(i)}) - f_{\hat\theta_{-j}}(\vec{x}^{(i)})]^2,
$$

where $p$ is the number of parameters $\theta$ and $\mathrm{MSE}$ is the mean squared error of the model.
For Cook's distance there are standard ways to plot the influence of individual points,
and a criterion to determine if the influence is statistically significant for linear regression.

The idea of Cook's distance can be extended beyond MSE and linear models.
The influence of deleting a training point $j$,
measured with a loss function $\mathcal{L}$, on the global dataset is

$$
\text{Influence}_{j} =
\mathcal{L}(\{z^{(k)}\}_{k\in\mathcal{E}},\hat\theta)]
- \mathcal{L}(\{\vec{x}^{(k)}\}_{k\in\mathcal{E}},\hat\theta_{-j}).
$$

The loss function to determine the influence is the same as for the optimisation here for simplicity, but this does not need to be the case.
Also, while for linear models the mean squared error loss
is usually computed on the training dataset $\mathcal{D}$
and underestimation is corrected with normalisation,
in the 
If the loss function $\mathcal{L}$ can be additively decomposed for each data point,

$$
\mathcal{L}(\{z^{(k)}\}_{k\in\mathcal{E}},\theta)] =
\frac{1}{|\mathcal{E}|} \sum_{k\in\mathcal{E}} \mathcal{L}(z^{(k)},\theta),
$$

then it is possible to also compute the influence of the deletion of $j$ on the $k$-th evaluation data point,

$$
\text{Influence}_j^{(k)} =
\mathcal{L}(\vec{x}^{(k)},\hat\theta)
- \mathcal{L}(\vec{x}^{(k)},\hat\theta_{-j}).
$$

These definitions of influence via deletion are exact and model-agnostic, but

1. They require retraining the model to obtain the influence of each training point, which can be prohibitively expensive, i.e. $\mathcal{O}(N)$, for large training datasets.
1. They assume that the optimal value of the hyperparameters is found precisely by the learning algorithm and consistent among retrainings, which is not the typical case for large models.

### Influence Functions

Influence functions approximate deletion influence using a gradient-based approach,
which eliminates the need for retraining.

The idea is to up-weight the $j$-th training data point by a small $\epsilon$

$$
\hat{\theta}_{\epsilon,j} = \arg\min_\theta[
\mathcal{L}(\{z^{(i)}\}_{i\in\mathcal{D}},\theta)
+ \epsilon \mathcal{L}(z^{(j)},\theta)
],
$$

and define the influence function as an infinitesimal version of the influence

$$
\text{IF}_j^{(k)} = \lim_{\epsilon\to 0}
\mathcal{L}(\vec{x}^{(k)},\hat\theta_{\epsilon,j})
- \mathcal{L}(\vec{x}^{(k)},\hat\theta).
$$

Expanding the influence function difference around $\hat\theta$ gives

$$
\mathrm{IF}_j^{(k)} =
\vec\nabla_\theta \mathcal{L}(z^{(k)},\hat\theta) \cdot
\frac{d\hat\theta_{\epsilon,z}}{d\epsilon}\bigg|_{\epsilon=0,z=z^{(j)}}.
$$

From the optimality condition of $\hat\theta$
and the positive definiteness of $\mathcal{L}$
it is possible to deduce

$$
\frac{d\hat\theta_{\epsilon,z}}{d\epsilon}\bigg|_{\epsilon=0}
= - H_{\hat\theta}^{-1} \cdot \vec\nabla_\theta\mathcal{L}(z,\hat\theta).
$$

The derivation is in the appendix of the original influence function paper.
Simple substitution then gives

$$
\mathrm{IF}_j^{(k)}
= -\vec\nabla_\theta\mathcal{L}(z^{(k)},\hat\theta)
\cdot H_{\hat\theta}^{-1} \cdot
\vec\nabla_\theta\mathcal{L}(z^{(j)},\hat\theta).
$$

One particularly interesting value is self-influence $j=k$,
which can be used to detect outlier or mislabelled data.


### Caveats

The assumptions to compute influence functions are:

1. That the loss function is differentiable two times.
2. That the Hessian $H_{\hat\theta}=\nabla_\theta\nabla_\theta\mathcal{L}$ is positive definite.
3. That the model parameters $\theta$ are at or near a local minimum.

Many overparametrised models have non-positive-definite Hessians.
One common solution is to add *damping*, i.e.

$$H \to H + \lambda I.$$

Also, direct inversion of $H_{\hat\theta}$ is computationally very heavy for models with many parameters.
This is why many variants use

- Hessian–vector products (HVPs)
- LiSSA
- Conjugate gradient methods  
- Low-rank curvature approximations (e.g., K-FAC, EK-FAC)


### Further comments

- Influence is also used to interpret Large Language Models, [paper](https://arxiv.org/abs/2308.03296) by Anthropic
- Influential subsets: groups of instances, complex
- Influence of a specific feature of a specific training data point is also possible!

### Resources

- Molnar's [chapter](https://christophm.github.io/interpretable-ml-book/influential.html) on influential instances
- `pyDVL` [page](https://pydvl.org/stable/influence/) on influence functions
- Summary [paper](https://arxiv.org/abs/2508.07297) with introductory explanations
- Position [paper](https://arxiv.org/abs/2501.18887) unifying data, feature, and component attributions
- `dattri` [library](https://github.com/TRAIS-Lab/dattri) and [paper](https://doi.org/10.52202/079017-4345)
