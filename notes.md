# Notes to self

## XAI Lecture 5: Feature Selection

Goals

- Faster computations
- Less data collection efforts
- Interpretability
- Generalization (curse of dimensionality / regularisation)

Applications

- Microarrays: genes
- Text analysis: words

Points to make

- Hypercube of size $2^N$ for feature presence or absence, exponential complexity.
- Univariate models and Leave-One-Out as two extremes for the search.
- Profile of forward selection or backward elimination as Pareto front.

## Absent features

The crux of the issue with feature attribution / importance is to define what happens when there is "no feature".
We usually do not have a model that can deal with that, and even if we do, this depends on the assumption.

- Is it a distinct value from all others?
- Is it a special value like 0?
- Is it a mean or median?
- Is it randomly drawn from the distribution?
- Is it taken from other points close to the instance of interest?
- Is it just omitting a token in a Transformer?
- Is it going down a random branch in a tree?
- Is it removing models from an ensemble?
- Is it a different model without the feature?
