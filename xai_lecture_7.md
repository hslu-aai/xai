# XAI Lecture 7: SHAP


## Shapley values

### Introduction

Shapley values were developed in **game theory and economics**.
They provide a way to **distribute a reward** (think about a profit)
among the people who contributed to its achievement,
respecting a certain formulation of **fairness**.

Because of this, the jargon of Shapley values is a bit special.
One considers a set $S = \\{1,2,\dots,n\\}$
of all $n$ possible **players** (or *workers*),
who cooperate in order to obtain a goal.
To figure out how to split the reward,
the idea is to consider a subset $S'\subset S$ of players called a **coalition**.
The **value** $v(S')$ that would have been obtained
by the players of each coalition
can be used for reward distribution.

```diff
! Example: Coalitions of three or four players

The possible coalitions of three players 1, 2, and 3 are
- 1x 0 players: {}
- 3x 1 player : {1}, {2}, {3}
- 3x 2 players: {1, 2}, {1, 3}, {2, 3}
- 1x 3 players: {1, 2, 3}

The possible coalitions of four players are
- 1x 0 players: {}
- 4x 1 player : {1}, {2}, {3}, {4}
- 6x 2 players: {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}
- 4x 3 players: {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}
- 1x 4 players: {1, 2, 3, 4}
```

```diff
! Question:

How many possible coalitions are there for a set of n players?
How many of these coalitions have exactly m players?
```

Each coalition corresponds to a binary vector of $n$ elements
which specifies if each element is present or not,
so it is an element of $\\{0, 1\\}^n$ and therefore there are $2^n$ possibilities.
Each coalition with exactly $m$ players
is equivalent to choosing $m$ elements out of $n$,
so there are exactly "$n$ *choose* $m$", or the binomial coefficient

$${n\choose m} = \frac{n!}{m!(n-m)!}.$$

To understand the value function $v$,
take the case of a delivery company with two riders 1 and 2,
and a marketing employee 3.
The company has no earnings when no employees work,
$v(\\{\\})=$ 0.
Suppose that one rider is faster than the other,
and the marketing employee does not earn anything alone
since they cannot ship anything.
Monthly earnings for individual coalitions could be
$v(\\{1\\})=$ 2'000,
$v(\\{2\\})=$ 3'000,
$v(\\{3\\})=$ 0.
If the two riders work together without a marketing agent,
they are not be able to find enough customers for both,
so $v(\\{1,2\\})=$ 4'000,
and alone with the marketing agent they cannot complete more deliveries,
$v(\\{1,3\\})=$ 2'000,
$v(\\{2,3\\})=$ 3'000.
When all three work together, they maximise monthly income
$v(\\{1,2,3\\})=$ 5'000.
In tabular form,

| Coalition | Value |
| :-: | --: |
| $\\{\\}$ | 0 |
| $\\{1\\}$ | 2'000 |
| $\\{2\\}$ | 3'000 |
| $\\{3\\}$ | 0 |
| $\\{1, 2\\}$ | 4'000 |
| $\\{1, 3\\}$ | 2'000 |
| $\\{2, 3\\}$ | 3'000 |
| $\\{1, 2, 3\\}$ | 5'000 |

How should the profit be distributed among them?


### Definition

The Shapley value for player $j$ is a weighted average
of the difference all coalitions that include this player
and all coalitions that do not include the player.
Written as a sum of all permutations of $n$ elements it reads

$$
\phi_j
= \frac{1}{n!} \sum_{\pi\in P_n}
[v(\\{k: \pi(k)<\pi(j)\\})-v(\\{k: \pi(k)\le\pi(j)\\})].
$$

The order of the sets within the value functions does not matter,
so this will have many equal terms in the sum.
Grouping them together
and denoting the set of all players that are not $j$
as $S_{-j} = S\setminus\\{j\\}$ one finds

$$
\phi_j
= \frac{1}{n} \sum_{S'\subset S_{-j}} \frac{1}{n-1\choose |S'|}
[v(S'\cup\\{j\\})-v(S')].
$$

For player 1 in the example before,
using the permutation formula we have

| Permutation | Coalition | Formula | Difference |
| --: | :-: | :-: | --: |
| 123 | $\\{\\}$ | $v(\\{1\\})-v(\\{\\})$ | 2'000
| 312 | $\\{3\\}$ | $v(\\{1,3\\})-v(\\{3\\})$ | 2'000
| 231 | $\\{2,3\\}$ | $v(\\{1,2,3\\})-v(\\{2,3\\})$ | 2'000
| 132 | $\\{\\}$ | $v(\\{1\\})-v(\\{\\})$ | 2'000
| 213 | $\\{2\\}$ | $v(\\{1,2\\})-v(\\{2\\})$ | 1'000
| 321 | $\\{2,3\\}$ | $v(\\{1,2,3\\})-v(\\{2,3\\})$ | 2'000

so $\phi_1=$ 1'833.

Using the subset formula we have

| Subset $S'$ | Weight | Formula | Difference |
| :-: | :-: | :-: | --: |
| $\\{\\}$ | 1 | $v(\\{1\\})-v(\\{\\})$ | 2'000
| $\\{2\\}$ | 1/2 |  $v(\\{1,2\\})-v(\\{2\\})$ | 1'000
| $\\{3\\}$ | 1/2 | $v(\\{1,3\\})-v(\\{3\\})$ | 2'000
| $\\{2,3\\}$ | 1 | $v(\\{1,2,3\\})-v(\\{2,3\\})$ | 2'000

which gives the same result with 4 terms instead of 6.

Similarly, we find $\phi_2=$ 2'833
and $\phi_3=$ 333.
We observe that $\phi_1+\phi_2+\phi_3=$ 5'000,
so the full amount is distributed among the players.

The gap between the number of terms in the two formulas
increases with the total number of players
(also called the size of the grand coalition).
Indeed, the permutation formula has complexity $n!\sim n^ne^{-n}$,
while the subset formula has complexity $2^n$.
Even if the second is better, this is impractical
for anything but the lowest numbers $n\lesssim 10$.

### Properties

One can show that Shapley values are the one and only way
to divide a reward that satisfies a set of intuitive properties.

- 1. **Efficiency**:
The whole reward is distributed among the players.

$$\sum_{i=1}^n\phi_i=v(\\{1,\dots,n\\}).$$

- 2. **Symmetry**:
Players that contribute equally to any coalition get the same reward.

$$
\phi_i=\phi_j
\qquad\text{if}\quad
v(S'\cup\\{i\\})=v(S'\cup\\{j\\})
\quad
\forall S'\subset S\setminus\\{i,j\\}.
$$

- 3. **Null effects**:
A player who never changes the value of a coalition gets no reward.

$$
\phi_j=0
\qquad\text{if}\quad
v(S'\cup\\{j\\})=v(S')
\quad
\forall S'\subset S\setminus\\{j\\}.
$$

- 4. **Linearity**:
When value functions are added,
the reward for each player is the sum of the original rewards.

$$
\phi_i[v_1+v_2]=\phi_i[v_1]+\phi_i[v_2].
$$

It is possible to show that properties 2, 3, and 4
can be replaced by a single but more involved requirement.

- [234] **Monotonicity**:
If the difference of a player according to a value function
is always greater or equal compared to a different value function,
then corresponding player reward must also be greater or equal.

$$
\phi_j[v_1]\ge\phi_j[v_2]
\qquad\text{if}\quad
v_1(S'\cup\\{j\\})-v_1(S')\ge v_2(S'\cup\\{j\\})-v_2(S')
\quad
\forall S'\subset S\setminus\\{j\\}.
$$

### Shapley feature importance

Shapley values can directly be used to assess global feature importance
in a way that encompasses both univariate and leave-one-feature-out estimates.
Given a trained model for each feature set $\hat{f}(\\{x_i\\}_{i\in S'})$
and a score function $s$, one simply sets

$$
v(S') = s(\hat{f}(\{x_i\}_{i\in S'}), y).
$$

Note that the number of players now corresponds to the number of features,
$n=|S|=D$.
The Shapley importance for feature $j$ then contains both the term for $S'=\\{\\}$,
which is the univariate feature importance,
and the term for $S'=S\setminus\\{j\\}$,
which is the leave-one-feature-out feature importance.
These two contributions have the same weight $1/D$.
In Shapley importance, they are also mixed with all the other feature set sizes
from $1$ until $D-1$, where each number of feature has the same total weight,
and is just the average of all possibile feature subsets
with that number of features.

Computing global Shapley feature importance requires to have a model
trained for each feature combination for a total of $2^D$,
which is totally unrealistic for anything but the simplest models.
Of course, one could also combine the bookkeeping of Shapley values
with other techniques such as permutation feature importance,
i.e., reshuffle not just one feature at a time,
but preserving only the subset of interest.
This does not require retraining, which may speed up the process significantly,
but does not get rid of the $2^D$ scaling
and suffers from unrealistic instances,
in other words neglected correlations and extrapolation.

To speed up evaluation, one can approximate Shapley values
by Monte Carlo sampling.
This is nothing else than randomly sampling many feature sets $S'$
and averaging over their contributions.
One improvement over naively including each feature or not
with 50% probability is to explicitly
stratify the sampling over the number of features,
which just means that an equal number of samples
are drawn for each target feature number from 0 until D-1.
The combinations with the least and most features
indeed have the least options and therefore the highest weight,
which makes the estimate more accurate.


## SHAP

SHAP, for SHapley Additive eXplanations, is not really a method
but a work that unified several similar XAI approaches under the same framework.
First published at NeurIPS in 2017, it picked up momentum
thanks to solid intuitive theory and a practical `python` package called `shap`.
It therefore also evolved and included more recent methods,
while updating choices in the old methods
according to experience gained by the community.

### Maskers

For model-agnostic explanations,
SHAP obtains value functions by turning features "on" and "off",
or in other words *masking* features.
For each inference operation,
a vector $\vec{z}\in\\{0,1\\}^D$ specifies
which features are used (1) or not (0).
There are several ways to achieve masking.

#### Independent

For tabular data, the absent features are set to their values
from a (random) sample in the (training) dataset,
and the procedure is repeated.
This is repeated for a certain number of samples
from 1 up to the dataset size to cover more values.

#### Text

For text data the initial formulations was a bag-of-words representation,
where the $i$-th word was omitted by simply setting its presence to $z_i=0$.
Now this is of course based on tokens,
which are set to the "mask" token if this is defined by the tokenizer
or to "..." otherwise.

#### Image

For image data, superpixels are set to be absent
either by blurring or inpainting using OpenCV.
Inpainting comes in two versions, one using Navier--Stokes equations (`ns`)
and another one based on fast marching (`telea`, from the name of the paper author).

#### Properties

```diff
! Question
Does this suffer from the issue of unrealistic points?
```

At least the procedures for image and tabular data
suffer from the issue of unrealistic points.
Image models were likely not trained with flat (or noisy) superpixels.
For tables, even if correlations among present features only
or absent features only are preserved,
dependencies across present and absent features are not respected.

### Explainers

#### Exact

When it is possible to sample all $2^D$ combinations,
potentially for many independent samples,
Shapley values can be computed using their defining formula.
SHAP suggests a number of features less or around 15 for this option.

#### Kernel

KernelSHAP is used when sampling all combinations of absent features is too slow.
KernelSHAP uses a 
the SHAP contributions for each feature
as the coefficients of a linear model
$$
g(\vec{z}) = \sum_{i=1}^D\phi_i z_i,
$$
where each variable  corresponds to the presence or the absence of a feature.

This is the same formula as LIME, except with different weights!
LIME attributes weights based on the distance to the instance of interest.
KernelSHAP uses the weights that would be attributed to Shapley values,
which depend on the number of active features.
However, this points to one further important difference.
The data space sampled in LIME is typically based
on Gaussian univariate distributions or uniform distributions per quantiles,
which changes all features at once.
The sampling in SHAP considers variations
depending on the number of present features.

Also, as it is the case for LIME,
it is common to generate **sparse** explanations
by using some kind of feature selection.
This can mean either of the following.

- Keeping a fixed number of features $k$,
  which means running KernelSHAP once, obtaining results,
  taking the top-$k$ features by SHAP absolute value,
  and repeating SHAP attribution only with these.
- Performing forward selection or backward elimination,
  stopping either when performance starts to decrease,
  or a certain number of features $k$ is reached,
  or using a different criterion
  (typically Akaike's or Bayes information criteria AIC/BIC).
- Using a version of LASSO to perform the linear fit,
  and discarding all features with zero coefficients.

#### Tree

A model-specific version of SHAP can be used
to speed up computations for tree-based models.
This applies to decision trees
but also to random forests or gradient boosting,
and to implementations such as XGBoost, LightGBM and CatBoost.

The trick is to let all the points flow through the tree at the same time.

### Plots

#### Waterfall plots

- Features from the most important (top) to the least important (bottom) according to the absolute SHAP value
- Start from the baseline at the bottom, marked on the horizontal axis, add up contributions one by one.
- Feature names shown on the left including values.
- The final value is marked with a grey line and shown on top.
- Contributions are arrows, where negative contributions are in blue pointing left, and positive ones are in red pointing right.
- The least important features may be grouped together and called "other", else the plot may become too tall with a lot of almost-zero contributions.
- Very good to visualize the effect of the most important features, which comes with clear sign and size.
- Visualizes *local* SHAP values.

![Waterfall plot](https://github.com/shap/shap/blob/master/docs/artwork/california_waterfall.png?raw=true)

#### Force plots

- Features are visualised all as a single series of arrows in a thick horizontal band.
- Positive importance features are to the left of the final value, negative ones on the right.
- More important features are closer to the final value.
- The base value and the final value are indicated on a horizontal axis parallel to the arrow line, together with a scale.
- The names and values of the most important features are reported below the thick arrow.
- The plot fits in a small vertical space, but individual feature effects are not so easy to read.
- Visualizes *local* SHAP values.

![Force plot](https://github.com/shap/shap/blob/master/docs/artwork/california_instance.png?raw=true)

- One can also combine force plots for many dataset points by rotating them vertically.
- Samples are sorted by similarity using hierarchical clustering for better visualisation.
- Visualizes *global* SHAP values, with individual resolution.

![Combined force plots](https://github.com/shap/shap/blob/master/docs/artwork/california_dataset.png?raw=true)

#### Scatter plots

- A scatter plot visualizes the SHAP value as a function of a feature value for all points in a dataset.
- It can be extended to include a second feature using colours as in the example, which can reveal interactions.
- Note the histogram on the horizontal axis to illustrate data coverage.
- Visualizes *global* SHAP values, with individual resolution.

![Scatter plot](https://github.com/shap/shap/blob/master/docs/artwork/california_scatter.png?raw=true)

#### Bar plots

- Plot the average magnitude of the SHAP attributions $\phi_j$.
- Gives some importance even to random features.
- Visualizes *global* SHAP values.

![Bar plot](https://github.com/shap/shap/blob/master/docs/artwork/california_global_bar.png?raw=true)

#### Beeswarm plots

- Plot the values of SHAP attributions for all features and all data points.
- On the horizontal axis the SHAP value is reported.
- The color indicates the feature value from low (blue) to high (red).
- One can see if the dependence of the outcome from the feature is monotonic or more complex.
- Visualizes *global* SHAP values, with individual resolution.

![Beeswarm plot](https://github.com/shap/shap/blob/master/docs/artwork/california_beeswarm.png?raw=true)

#### Decision plots

![]

#### Text plots

## Additional references

Feel free to consult:

- The [documentation](https://shap.readthedocs.io) of the `shap` python package, and in particular the [overviews](https://shap.readthedocs.io/en/latest/overviews.html).
