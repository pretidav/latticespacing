---
layout: post
title: "Random walks on a lattice"
category: physics
---
A quick example post on a classic: the simple random walk on $\mathbb{Z}$.

## Setup

At each step the walker moves $+1$ or $-1$ with equal probability. After $n$ steps, the position is

$$
X_n = \sum_{k=1}^{n} \xi_k, \qquad \xi_k \in \{-1, +1\}.
$$

The mean is $\langle X_n \rangle = 0$ and the variance grows linearly:

$$
\langle X_n^2 \rangle = n.
$$

So the typical distance from the origin scales as $\sqrt{n}$, which is the signature of diffusion, $\langle x^2 \rangle = 2Dt$.

## Distribution

The probability of being at site $x$ after $n$ steps is binomial, and for large $n$ it approaches a Gaussian:

$$
P(x, n) \approx \frac{1}{\sqrt{2\pi n}} \, e^{-x^2 / 2n}.
$$

## Simulation

```python
import numpy as np

n, walkers = 1000, 10_000
steps = np.random.choice([-1, 1], size=(walkers, n))
x = steps.sum(axis=1)
print(x.var())  # close to n
```

> Put a plot in `assets/img/2026-11-18-random-walks/` and include it with the figure helper.
