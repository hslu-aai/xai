# Lecture 14: Recent Developments

## Legal Requirements and XAI Methods

Please consult the slides on ILIAS, which are based on 

### References

- Sovrano, F., Vilone, G., Lognoul, M., & Longo, L. (2025). Legal XAI: a Systematic Review and Interdisciplinary Mapping of XAI and EU Law. DOI: <https://doi.org/10.2139/ssrn.5371124>

## Mechanistic Interpretability

### Introduction

- Most XAI methods seen are **model-agnostic**
  - The model is treated as a black box, i.e., only inputs and outputs are observed
- Some of the methods considered **partial aspects** of the models' inner workings

**Mechanistic interpretability**, in short MechInterp, is the ambitious program
of **reverse-engineering** entirely the **inner workings of a model**,
decomposing it into all of its computational units or circuits,
and understanding the algorithms it implements.

This program is characterised by

- An insistence on **real models**, not toy models, which sets a focus on **transformers** and LLMs
- The goal of improving **AI safety**
  - Some models exhibit forms of *deception*
  - Some models apparently try to avoid being *shut down* (in this case it was sufficient to read the chain-of-thought)
- A strong industry drive mostly by Anthropic and Google
- Here **introduction only**

### Transformers Recap

Transformers take as input a **sequence** $X$ of tokens $t_i$ with embeddings $x_i$, including **causal constraints**.
They are characterised by three main ingredients.

#### (1) Feedforward Network (FFN)

Also known as **Multilayer Perceptron (MLP)**.

$$
\text{FFN}(x_i) = g(x_i W_{in}) W_{out}
$$

- Following some modern literature, biases are omitted (they are actually missing in some architecture)
- $x_i W_{in}$ is a linear combination
- $g()$ applies a non linearity
- $g(x_i W_{in})$ altogether is known as a "neuron"
- Multiplication by $W_{out}$ applies another linear combination

#### (2) Attention Layer

Each head $h$ computes the output

$$
A^h(X_{\le i}) = \sum_{j\le i} a_{ij}^h x_j W_V^h W_O^h,
$$

- $W_V$ is the *value* matrix
- $W_O$ is the *output* matrix
- The product $W_VW_O\equiv W_{VO}$ is the value-output matrix

Attention weights:
$$
a_{ij}^h = \text{softmax}
\frac{x_i W_Q^h W_K^h X_{\le i}}{\sqrt{d_h}}
$$

- $W_Q$ is the *query* matrix
- $W_K$ is the *key* matrix
- The product $W_QW_K\equiv W_{QK}$ is the query-key matrix

#### (3) Residual Stream

A classical MLP computes

$$
\text{FFN}_N(\text{FFN}_{N-1}(\dots \text{FFN}_1(x)))
$$

A Residual Network (ResNet) computes

$$
\text{FFN}_N(\text{FFN}_{N-1}(\dots \text{FFN}_1(x)))
\;+\; \text{FFN}_{N-1}(\dots \text{FFN}_1(x))
\;+\; \dots
\;+\; \text{FFN}_1(x)
$$

So the overall picture is [TODO figure]

- Alternating **ATT** and **FFN**
- Output from **residual stream**


#### (1+2+3) Sum-of-Paths Picture

- Output = sum over many **paths**
  - Attention paths
  - FFN paths
  - Residual connections

> **Surprising fact**  
> Removing even a **large number of attention heads**  
> may *not* change the output in any significant way


### Techniques

#### (A) Activation Patching

Change the value of a **computed variable** in the **forward pass**.

Example: zero-patching  

```diff
! Question
What is wrong with this?
```

- It may probe the model out of distribution
- It may generate invalid inputs

Better options are

- Noise (add Gaussian)
- Resample (take value from other data points)
- Mean (take the mean from a set of data points)

The idea essentially copies the **Word2Vec algebra**

$$
\text{Paris} - \text{France} + \text{Italy} = \text{Rome}
$$

Example: Golden Gate Claude

#### (B) Probes

Take a **layer**, explore if it has learned something

$$
p(x_i^\ell) = z
$$

for given $x_i^\ell$ and $z$, where $z$ is the "something" the model should have learned

```diff
! Question
Do you already know how to do this?
```

Concept vectors!

$$
b + W x_i^\ell = z
$$

Crucially, this is a **linear** probe

```diff
! Question
Why not build a more complicated probe?
```

Else the probe can **learn to extract the concept itself**
Actually, even for transformers, there is a lot of theory that confirms **linear separability**.

Applications of probes are

- Malignant prompt / jailbreak detection  
  - Model may produce outputs even if it *knows* it's being manipulated!
  - Important: this is cheap, so applicable in production
- Deception detection

Problems with probing are

- Works if we **know what we are looking for**
- Looking at neurons is not great because they suffer from **superposition**
  - Concepts conflated

#### (C) Sparse Autoencoders (SAEs)

- The **standard** autoencoder (AE) works by compressing the input to a low number of dimensions (creating a *bottleneck*) and then reconstructing it.
The layer with low dimensions is interpreted as a representation of the input.
- An **overcomplete** AE actually expands the input to a **higher** number of dimensions, but the surprising thing is that this still learns, i.e., it produces meaningful representations because of learning dynamics.
- A **sparse** AE is an overcomplete one with a very high $L_1$ loss. This is connected to **dictionary** learning, as the model now has pressure to learn **one thing at a time**.

[TODO figure]

SAEs in MechInterp can be very large, with up to **100M neurons**.

### Circuits Discovered

#### Modular addition

- $(1 + 3) \mod 3 = 1$
- $(1 + 5) \mod 3 = 0$

The algorithm implemented by a transformer is

1. Turn numbers into circles
1. Rotate circles  
1. Find sum  
1. Use multiple circles with different frequencies to resolve numerical ambiguities!!

[TODO figure]

#### Induction circuits

- `A B C D C D A ?`
- This works at other levels too → with concepts

#### Other circuits

- Copy
- Suppression

## Further reading

- Ferrando, J., Sarti, G., Bisazza, A., & Costa-Jussà, M. R. (2024). A primer on the inner workings of transformer-based language models. DOI: <https://doi.org/10.48550/arXiv.2405.00208>
- Nanda, N. (2025). How to become a mechanistic interpretability researcher. [Blog](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)
