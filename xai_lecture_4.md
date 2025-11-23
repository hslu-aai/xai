# XAI Lecture 4: Feature Importance


## Introduction

Feature importance is one of the **most requested** analyses in applied AI.
This reflects a basic explainability need: stakeholders want to know what are relevant factors for AI systems, as this influences their trust but also downstream decisions.
Typically, this is more about getting information about the **physical world**, rather than analysing a specific model.

```diff
! Repetition exercise:

What are the implications of targeting physical world insights
for producing feature importance estimates?
```

Therefore, feature importance values usually have to be interpreted according to the *performance* of the model on unseen data, which sets a lower bound for their actual impact (if a model can use them to obtain a certain performance, they might still be suboptimally exploited, but they can at least achieve that level of accuracy).
Also, most of the time people talk about feature importance **globally** over the whole dataset rather than on a single instance.
This means that feature importance should summarise the average effect on data points, taking into account frequent but weaker influence as well as rare but stronger infuence.

### Input terminology

Before embarking on the technical discussion of feature importance methods, it is useful to set some terminology.

- **Variables** are in principle any quantities that change value, but the term in ML typically refers to the entries of a dataset that change according to the sample (i.e., the columns for tabular data). These are close to the raw values collected.
- **Features** in ML are the inputs that are directly used by a model. Compared to variables, they may have been subject to transformations that range from simple normalisation to interaction term generation or more complex operations.
- **Covariates** are variables that may counfound the effect of a treatment in statistical modelling. In the community of statisticians, including books and code, this term is sometimes used instead of features or variables.

Given a dataset with $N$ samples and a set of features $S$, we say that a feature $F$ is

- **Strongly relevant** within $S$ if removing the feature $F$ will result in a suboptimal decision for at least one sample, i.e., it will decrease the performance of an optimal classifier
$$
\exists i:\quad p(y|x_1^{(i)},\dots,x_n^{(i)})\ne p(y|x_1^{(i)},\dots,x_{j-1}^{(i)},x_{j+1}^{(i)},\dots,x_n^{(i)}).
$$
- **Weakly relevant** within $S$ if there is at least one subset $S'\subset S$ such that $F$ is strongly relevant within $S'$, i.e., removing the feature may decrease performance, depending on other features. In this case, one also says that the $F$ is **redundant** within $S$ (but not within $S'$).
- **Irrelevant** otherwise, i.e., removing the feature never decreases performance.

By contrast, the concept of **usefulness** of a feature is connected to whether this is actually used by the model in a way that improves the prediction

### Method taxonomy

There are three main classes of feature importance methods:

- **Embedded** methods are model-specific, as they require access to the inner workings of a model to estimate its importance.
- **Filter** methods are model-agnostic procedures that do not require access even to model inference to obtain importance estimates.
- **Wrapper** methods are also model-agnostic, but need to be able to run inference with the model and sometimes even retrain it.

### Result format

The results of feature importance and selection can take different forms that are connected to each other.

- **Scores** are continuous variables that indicate the importance of each feature.
- A **rank** is a sorted lists where more important features appear earlier, and is therefore equivalent to carefully chosen ordinal scores.
- A **subset** is simply an unsorted list of the features that are considered to be important.

Note that it is always possible to rank features according to scores by sorting them, and to select a subset by choosing a threshold in the ranks or scores.
The other direction (obtaining scores from ranked features or ranked features from a subset), however, is not in general a valid operation.

## Embedded Methods

Here there is only a short recap of embedded methods, since many were discussed in the lecture on intrinsically interpretable models.

### Linear Importance

For linear models, two ways to measure feature importance are:

- The size of the coefficients times the standard deviation or interquantile range of the feature. If features are normalized, multiplying by a measure for the feature variation is not necessary.
- The $Z$-score of the coefficients, i.e., the number of standard deviations they differ from zero, where the standard deviation is obtained during the fit (based on the Hessian at the minimum).

### Tree-based Importance

For tree-based models, some ways to measure feature importance are:

- Counting the number of splits a certain feature is used.
- Adding the decreases in impurity or entropy produced by all splits involving the feature.

For ensemble models with bagging, e.g., random forests, weak learners are trained with different feature subsets.
In this case there is also the option to run evaluation discarding all weak learners that use a certain feature, and report the score difference of the new sub-ensemble.

## Filter Methods

The idea is to find features which have proven correlations with the target.

> **Digression:** Permutations
> 
> Permutations are bijective functions that send a finite element set into itself.
> 
> $$
> \pi: e\in F\to \pi(e)\in F,\qquad\pi(F)=F.
> $$
> 
> Without loss of generality, the elements of $F$ can be numbered $1$ through $N$.
> Therefore, permutations can be uniquely identified by a sequence of the frst $N$ numbers which specify the elements associated to each of $1,2,\dots,N$.
> To count how many there are, consider that the first element is to be picked among $N$ choices, the second only has $N-1$, and so on, so in total there are $N!$ permutations of $N$ elements.
> 
> One can define the **inversion number** of a permutation which is the number of swaps that are needed to sort its elements.
> This can be determined graphically by listing the sequences before and after the permutation, joining equal elements with lines, and counting how many times they cross.
> Finally, the **sign** of a permutation is $-1$ if the inversion number is odd and $+1$ if it is even.

### Correlation Coefficients

The most common correlation coefficient is Pearson's $r$, which quantifies the linear dependence between two variables

$$
r(x,y) := \frac{
\langle\bar x\bar y\rangle
}{
\sqrt{\langle\bar{x}^2\rangle\langle\bar{y}^2\rangle}
}.
$$

where $\bar{x} = x - \langle x\rangle$ and $\bar{y} = y - \langle y\rangle$.
In practice, the expectation value $\langle\cdot\rangle$ is obtained by averaging over all data points $i$

$$
\langle z\rangle = \frac{1}{N}\sum_{i=1}^N z^{(i)},
$$

and $r$ can be interpreted as the cosine of the angle between the vectors $\bar{x}^{(i)}$ and $\bar{y}^{(i)}$, since

$$
r = \frac{\bar{x}\cdot\bar{y}}{|\bar{x}||\bar{y}|}.
$$

In other words, if the vectors $x$ and $y$ have zero mean and unit standard deviation, one simply has $r=x\cdot y$.
The Pearson correlation coefficient equals +1 when there is a linear dependence $y=ax+b$ with $a$ positive, and -1 when there is a linear dependence with $a$ negative:

$$
r(x,y) = \frac{
\langle(x - \langle x\rangle)(ax-\langle ax\rangle)\rangle
}{
\sqrt{\langle (x-\langle x\rangle)^2\rangle}
\sqrt{\langle (ax-\langle ax\rangle)^2\rangle}
} = \frac{a}{|a|} = \mathrm{sign}(a).
$$

Example of data distributions which result in zero Pearson correlation coefficient are rotationally symmetric ones.
Elliptic ones give the main axis.
It is sufficient to have rotational symmetry by 90 degrees rotations or mirror symmetry with respect to one of the axes: for each point $(x, y)$, there is a point $(y, -x)$ or $(x, -y)$ or $(-x, y)$ that cancels its contribution.
The so-called coefficient of determination $r^2$ is also what is often used to quantify the fraction of variance explained by a linear model.

For relations that may not be linear, one option is to use **Spearman**'s $\rho$, which is defined as the Pearson correlation coefficients between ranks.
If we define $R(x)$ to be the rank of the items in $x$, i.e., the vector of their positions in sorted order, we have

$$\rho := r(R(x), R(y))$$

This is equal to $+1$ or $-1$ for any variables whose dependence on each other is monotonic.

A different option to measure the interaction of two features is **Kendall**'s $\tau$, which is defined as the number of element pairs that are sorted the same way in $x$ and $y$ (concordant), minus the number of pairs that are sorted opposite (discordant).
An equation that defines Kendall's correlation coefficient is

$$
\tau := \frac{\sum_{i,j}\mathrm{sign}(y_i-y_j)}{n(n-1)/2}.
$$

This can be obtained graphically by drawing the permutation of the ranking, counting the number of swaps, i.e. the permutation's inversion number.

All of these are special cases of a **general correlation coefficient** which can be formulated in matrix form for antisymmetric matrices $A_{ij} = -A_{ji}$, i.e. $A=-A^\mathrm{T}$.
This reads

$$
c(X, Y) := \frac{X\cdot Y}{|X||Y|},
$$

where $X\cdot Y = \sum_{ij}X_{ij}Y_{ij}$ is the Frobenius dot product and $|A|=\sqrt{A\cdot A}$ the Frobenius norm.
Then we have

- $c = r$ when $A_{ij} = a_i - a_j$,
- $c = \rho$ when $A_{ij} = R(a)_i - R(a)_j$,
- $c = \tau$ when $A_{ij} = \mathrm{sign}(a_i - a_j)$.

> **Example: $y = x^3$**
>
> | x | y | xx | yy | xy | R(x) | R(y) |
> | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
> | -2 | -8 | 4 | 64 | 16 | 1 | 1 |
> | -1 | -1 | 1 | 1 | 1 | 2 | 2 |
> | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
> | +1 | +1 | 1 | 1 | 1 | 4 | 4 |
> | +2 | +8 | 4 | 64 | 16 | 5 | 5 | 
> | --- | --- | --- | --- | --- | --- | --- |
> | 0 | 0 | 10 | 130 | 34 | 15 | 15 |
> 
> $$
> r = \frac{34}{\sqrt{1300}} = 0.943,\qquad\rho=+1,\qquad\tau=+1.
> $$

> **Example: $y = x^2$**
>
> | x | y | xx | yy | xy | R(x) | R(y) | $\bar{R}(x)$ | R(y) |
> | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
> | -2 | 4 | 4 | 16 | -8 | 1 | 1.5 | -2 | -1.5 |
> | -1 | 1 | 1 | 1 | -1 | 2 | 3.5 | -1 | 0.5 |
> | 0 | 0 | 0 | 0 | 0 | 3 | 5 | 0 | 2 |
> | +1 | 1 | 1 | 1 | +1 | 4 | 3.5 | 1 | 0.5 |
> | +2 | 4 | 4 | 16 | +8 | 5 | 1.5 | 2 | -1.5 |
> | --- | --- | --- | --- | --- | --- | --- |
> | 0 | 10 | 10 | 34 | 0 | 15 | 15 |
> 
> $$
> r = \frac{0}{\sqrt{340}} = 0,\qquad\rho=0,\qquad\tau=0.
> $$

-

> **Example: ranking**
>
> | x | y | xx | yy | xy |
> | :-: | :-: | :-: | :-: | :-: |
> | 1 | 2 | 1 | 4 | 2 |
> | 2 | 1 | 4 | 1 | 2 |
> | 3 | 3 | 9 | 9 | 9 |
> | 4 | 5 | 16 | 25 | 20 |
> | 5 | 4 | 25 | 16 | 20 |
> | --- | --- | --- | --- | --- |
> | 15 | 15 | 55 | 55 | 53 |
> 
> $$
> r = \rho = \frac{53/5-9}{8} = 0.2,\qquad\tau=0.6.
> $$

 
One way to quantify feature importance is to use $|c(x_j,y)|$ between a feature and a target, because a nonzero correlation guarantees that the feature reveals information about the target up to statistical fluctuations.
However, the opposite is not necessarily true, since there are patterns that give zero or small correlations despite a relation between observables.

One very tricky issue is that the real correlations are usually unknown, and one has to resort to empirical estimates which use a finite sample size.
The resulting statistical fluctuations may therefore create spurious correlations that result in wrong feature selection.
This problem is exhacerbated when using a large number of features, since the chance to accidentally find a large correlation adds up for each feature and becomes significant.
This could in principle be mitigated or corrected.

### Mutual Information

Mutual information between probability distributions is defined by

$$
\mathrm{MI}(j) = \mathrm{D}_{\mathrm{KL}}(p(x,y)\|p(x)p(y))
= \sum_x\sum_y p(x, y) \log\frac{p(x,y)}{p(x)p(y)}.
$$

This looks much better than a correlation, because it can capture really non-trivial relationships such as the ones with vertical mirror symmetry.

> **Example: y=x^2 again**
> 
> Assuming points with probability 1/5, we have
> $$
> \mathrm{MI}(j) = 4\times\frac{1}{5}\log\frac{1/5}{1/5\,2/5} + \frac{1}{5}\log\frac{1/5}{1/5\,1/5}
> = \log 5-\frac{4}{5}\log 2.
> $$

The issue is that as soon as the variables have unique values per point, mutual information is always maximal.
This is the case for continuous features or targets, as well as for high-dimensional spaces (not the case here, since each variable is considered indepedently).

In practice, this means that the probability densities for continuous variables have to be estimated and this opens Pandora's box.
In practice this can be done with histograms, or with Kernel Density Estimation (KDE), but setting parameters is not trivial and changes the results significantly.
Using a resolution that is too high gives back the case where each value is considered a point, and using a resolution that is too low yields a single distribution, e.g., a Gaussian recovers Pearson's correlation coefficient which is not very useful.

### Univariate Models

An option along these lines is to fit a model to the data using one feature at a time only, and use a score to quantify the performance gain compared to using no features.
Using no features at all means that there is no way to distinguish data points, so the prediction must be constant or random.

The question in this case is which model and score to choose.
In fact, picking the coefficient of determination $r^2$ and the a linear model is equivalent to using Pearson's $r$ as a filter method.
This actually goes very much in the direction of a wrapper method.
The distinction between filter and wrapper methods was not that sharp after all.
The rough idea is that when models and scores are sufficiently simple, feature importance is more of a property of the data rather than a property of the specific models and scores themselves.

## Wrapper Methods

### Permutation Feature Importance

This method and the ones that follow it all require a score or metric $s$, which may also be the same as the loss function used for training.
The idea is that the importance of a feature $j$ can be estimated by destroying its relation with the target and analizing model performance.
The set of all values taken by the feature over the dataset is a good approximation of its generating distribution.
The relation between feature $j$ and the target is destroyed by permuting, i.e. shuffling that feature across the dataset as follows,
$$
\text{P-FI}(j) = s\big(y^{(i)},\hat{f}(x^{(i)}_1,\dots,x^{(i)}_D)\big)
	- s\big(y^{(i)},\hat{f}(x^{(i)}_1,\dots,x^{\pi(i)}_j,\dots,x^{(i)}_D)\big),
$$
Since some scores are better when they take higher values, and some are better when they take lower values, it is usually convenient to change the signs in this equation as appropriate to ease interpretation.
This way, positive scores indicate important features, and negative scores mean that features are detrimental, which essentially only happens on evaluation sets.
Importances close to zero, instead, indicate that the features are not relevant or not used by the model.
For instance, when a feature is repeated two times, each of the two copies may have a different performance depending of what the model learned specifically.

When a single permutation $\pi$ of the feature is chosen, there is a chanced that the shuffling is particularly lucky or unlucky, and the feature importance score is not reliable.
To mitigate this, it is possible to draw many permutations and report both their mean or median and 95% Confidence Interval.

```diff
! Exercise: Permutation Importance by Hand.

Compute permutation feature importance by hand
for the model $y=x^2 + 1$ on the dataset

|  x  |  y  |
| --- | --- |
| 0.0 | 0.9 |
| 0.9 | 2.8 |

```

Permutation feature importance is elegant and does not require retraining the model, which makes it fast compared to other alternatives.
However, it badly suffers from extrapolation and unrealistic data points as we saw for Ceteris Paribus.
Also, it is important to decide and report if the analysis is carried out on the training set, where the baseline performance score may be inflated due to overfitting, or on an evaluation data set.

### Leave One Feature Out

To avoid evaluating models based on data points unlike anything they were ever trained for, one option is to retrain them explicitly without the feature whose importance is to be estimated.
The score difference can then be used as an estimate of importance,
$$
\text{LOO-FI}(j) = s\big(y^{(i)},\hat{f}(x^{(i)}_1,\dots,x^{(i)}_D)\big)
	- s\big(y^{(i)},\hat{f}_{\!r}(x^{(i)}_1,\dots,\text{no}\ x^{(i)}_j,\dots,x^{(i)}_D)\big).
$$
This method is called *leave-one-out* (LOO), or *leave-one-feature-out* (LOFO).

LOFO does not evaluate models using unrealistic data points, which makes it considerably more solid than permutation feature importance.
It also has a straightforward interpretation, as it answers the question of what the model performance would be if the feature were not available.
Running LOFO however requires retraining, which may not be possible in some cases and is considerably slower compared to permutations.
It also compares the behaviour of *different models*, so it is not great for drawing conclusions about a single model---it gives information about the class of model on the given problem.
One further critique that was raised against LOFO is that it compares models trained with different number of features, and performance might be systematically higher when more features are available.
This issue mostly relates to the case where a model with high capacity is trained and evaluated on the same split, so it can be mitigated with an evaluation set---or with the next method.

### Permute and Relearn

To avoid systematic biases due to the comparison of models with a different number of features, it is possible to combine permutation feature importance with leave-one-feature-out.
This is done by permuting a feature, retraining the model, and computing the score difference, a method known as permute-and-relearn:
$$
\text{PAR-FI}(j) = s\big(y^{(i)},\hat{f}(x^{(i)}_1,\dots,x^{(i)}_D)\big)
	- s\big(y^{(i)},\hat{f}_{\!r}(x^{(i)}_1,\dots,\text{no}\ x^{(i)}_j,\dots,x^{(i)}_D)\big).
$$

Like LOFO, this requires retraining the model and is computationally expensive.
Like permutation feature importance, it produces data points which are unrealistic.
However, in this case the model *sees these data points during training*, and therefore it is not extrapolating in an uncontrolled fashion during inference.
Like LOFO, permute-and-relearn compares *different models*, so it is not great for drawing conclusions about a single model.
This technique is less common than permutation feature importance and LOFO, but it is a straightforward extension which might provide a useful check.

## Exercises

### Exercise 4.1: Wrapper methods differences

Train a regression model on a training split of the California housing dataset, and compute permutation feature importance, LOO feature importance, and permute-and-relearn feature importance *on the training split*.
Which differences can you observe across methods?

Now repeat the calculation, but using the validation split.
Which methods change their estimate of feature importance the most when changing the data split?
Do the differences across methods on the validation split stay the same?

### Exercise 4.2: Embedded vs wrapper methods

Train a linear regression model on the California housing dataset.
Obtain feature importance using the embedded method of Z scores and the wrapper method LOFO on the validation set.
Plot results as a bar plot next to each other, and compare differences.
