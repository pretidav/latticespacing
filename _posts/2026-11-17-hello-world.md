---
layout: post
title: "Hello, lattice"
category: meta
---
First post. This one checks that math and code render.

Inline math: Euler's identity is $e^{i\pi} + 1 = 0$.

Display math:

$$
Z = \sum_{\{s\}} e^{-\beta H(s)}, \qquad H(s) = -J \sum_{\langle i,j \rangle} s_i s_j
$$

And some code:

```python
import numpy as np
spins = np.random.choice([-1, 1], size=(16, 16))
```
