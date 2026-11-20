---
layout: post
title: "Go Your Ways, and Pour Out the Vials of the Neural Gas upon the Earth"
category: ml
---
> A Growing Neural Gas (GNG) is a graph that crawls onto a point cloud: units sit on the data, edges say which units are neighbours, and a new unit is born wherever the quantization error piles up. There is no schedule and no target size fixed in advance. I implemented the algorithm of Fritzke [1] in Python, and in this post I derive what each step of it does and then show how the code follows it. Let me try to formalize it.

##### Setting

Let $\xi \in \mathbb{R}^n$ be a random input signal with law $P$, and let signals $\xi_1, \xi_2, \dots$ be iid draws from $P$. The network consists of

- a finite set $\mathcal{A}$ of units, each with a reference vector $w_c \in \mathbb{R}^n$ (its position in input space);
- a set $\mathcal{N}$ of undirected edges between units. Edges are unweighted: their only job is to define a topology. Each edge carries an integer age, and each unit carries a non-negative error variable $E_c$.

The neighbours of $c$ are $\mathcal{N}(c) = \lbrace c' : \lbrace c, c' \rbrace \in \mathcal{N} \rbrace$. The Voronoi cell of $c$ and the winner of a signal are

$$V_c = \lbrace \xi : \lVert \xi - w_c \rVert \leq \lVert \xi - w_{c'} \rVert \ \ \forall c' \in \mathcal{A} \rbrace, \qquad s_1(\xi) = \arg\min_{c} \lVert \xi - w_c \rVert^2,$$

and the runner-up is $s_2(\xi) = \arg\min_{c \neq s_1} \lVert \xi - w_c \rVert^2$. In what follows I assume that $P$ has a bounded density and a finite second moment, and that the units sit at distinct positions, so that ties have probability zero and $s_1, s_2$ are a.s. well defined.

The picture below shows it for nine units in the plane. The cell $V_c$ is the region of the plane that is closer to the unit $c$ than to any other unit: the borders between cells are the perpendicular bisectors of pairs of units. A signal $\xi$ falls in exactly one cell, and the unit of that cell is its winner $s_1$. The runner-up $s_2$ is the second-closest unit, and its cell always shares a border with the one of the winner. This is why the edges that GNG draws between $s_1$ and $s_2$ form a subgraph of the Delaunay triangulation [1].

{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/voronoi.png" caption="Voronoi cells of nine units. The shaded cell is the one of the winner s1 of the signal ξ (black square, solid line). The runner-up s2 is the second-closest unit (dashed line)." %}

Two quantities will matter. The local contribution of unit $c$ to the quantization error, and the probability that a pair of units are the two closest ones to a signal:

$$D_c = \int_{V_c} \lVert \xi - w_c \rVert^2 \, P(\xi) \, d\xi, \qquad \rho_{cc'} = \Pr\big[ \lbrace s_1, s_2 \rbrace = \lbrace c, c' \rbrace \big].$$

The total quantization error is $\mathcal{E} = \sum_c D_c$.

Two more notions will be needed. The *Delaunay triangulation* of the units connects every pair of units whose Voronoi cells share a border: in the picture above, the units whose cells touch. It depends only on where the units are. The *induced Delaunay triangulation* of Martinetz and Schulten [2, 3] keeps only the pairs whose common border lies, at least partly, where the data are, that is, where $P > 0$ [1]. Equivalently, it keeps the pairs with $\rho_{cc'} > 0$: there are signals for which $c$ and $c'$ are the two closest units. The data act as a mask. For units on a ring, the cells across the empty interior touch, so the Delaunay triangulation would draw chords through the middle of the ring, while the induced one only joins the neighbours along the ring. Martinetz [4] showed that the induced triangulation preserves the topology of the data optimally, in a very general sense. This is the structure that GNG is built to learn.

##### The algorithm

The steps of the paper [1] are the following. The parameters are $\lambda \in \mathbb{N}$, $\varepsilon_b, \varepsilon_n \in (0,1)$, $\alpha \in (0,1)$, $d \in (0,1)$ and $a_{\max} \in \mathbb{N}$.

- **0.** Start with two units at random positions in $\mathbb{R}^n$. (The code also joins them by an edge, which the first signal would do anyway in step 6.)
- **1.** Draw a signal $\xi \sim P$.
- **2.** Find the winner $s_1$ and the runner-up $s_2$.
- **3.** Increment the age of all edges emanating from $s_1$.
- **4.** Add the squared distance to the error of the winner: $E_{s_1} \leftarrow E_{s_1} + \lVert \xi - w_{s_1} \rVert^2$.
- **5.** Move the winner and its direct neighbours towards the signal: $w_{s_1} \leftarrow w_{s_1} + \varepsilon_b (\xi - w_{s_1})$, and $w_{m} \leftarrow w_{m} + \varepsilon_n (\xi - w_{m})$ for all $m \in \mathcal{N}(s_1)$.
- **6.** If $s_1$ and $s_2$ are joined by an edge, reset its age to $0$. Otherwise create it, with age $0$.
- **7.** Remove the edges with age larger than $a_{\max}$, and then the units left without edges.
- **8.** If the number of signals so far is a multiple of $\lambda$, insert a unit:
   - $q = \arg\max_c E_c$, and $f = \arg\max_{c \in \mathcal{N}(q)} E_c$;
   - add a unit $r$ with $w_r = \tfrac{1}{2}(w_q + w_f)$;
   - replace the edge $\lbrace q, f \rbrace$ by the two edges $\lbrace q, r \rbrace$ and $\lbrace r, f \rbrace$;
   - $E_q \leftarrow \alpha E_q$, $E_f \leftarrow \alpha E_f$, and $E_r \leftarrow E_q$ (the new value).
- **9.** Decay all errors: $E_c \leftarrow d \, E_c$.
- **10.** Go to step 1 unless a stopping criterion (net size, or an error threshold) is met.

All parameters are constant in time. The values used in the paper, and in my examples, are $\lambda = 100$, $\varepsilon_b = 0.2$, $\varepsilon_n = 0.006$, $\alpha = 0.5$, $a_{\max} = 50$, $d = 0.995$.

> Let's prove what the steps actually do. Throughout, $\mathbb{E}$ is the expectation over the next signal, conditional on the current state of the network.

##### Steps 2 and 5: online quantization

Consider a unit $c$. It moves by $\varepsilon_b(\xi - w_c)$ when $\xi \in V_c$, and by $\varepsilon_n(\xi - w_c)$ when $\xi \in V_{c'}$ for a neighbour $c' \in \mathcal{N}(c)$. Taking the expectation over $\xi$,

$$\mathbb{E}[\Delta w_c] = \varepsilon_b \int_{V_c} (\xi - w_c) \, P(\xi) \, d\xi + \varepsilon_n \sum_{c' \in \mathcal{N}(c)} \int_{V_{c'}} (\xi - w_c) \, P(\xi) \, d\xi.$$

Now look at the first term alone. By definition, the quantization error is the sum over the cells of the error made in each cell:

$$\mathcal{E}(w) = \sum_k D_k = \sum_k \int_{V_k} \lVert \xi - w_k \rVert^2 \, P(\xi) \, d\xi.$$

Move the unit $c$ from $w_c$ to $w_c + \delta$, where $\delta$ is a small vector, and keep the other units where they are. Two things change: the squared distance to the unit $c$, and the cells, because the border between $V_c$ and a neighbouring cell $V_{c'}$ is the perpendicular bisector of the two units. The distribution $P(\xi)$ does not change: the data do not depend on the units. Write $g_k(\xi) = \lVert \xi - w_k \rVert^2$ for the squared distances before the move, and $g_k'$ after it, so that

$$g_c'(\xi) = \lVert \xi - w_c - \delta \rVert^2, \qquad g_k' = g_k \quad (k \neq c).$$

{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/shift.png" caption="The unit c moves by δ (exaggerated). The old border of its cell is dashed and the new borders are solid. The strips between them hold the signals that change owner: the orange one passes to c, the grey ones leave c. The width of the strips is proportional to δ." %}

Call $V_k'$ the cells after the move, so that $\mathcal{E}' = \sum_k \int_{V_k'} g_k' \, P \, d\xi$. Split the change of the error in two parts: the new integrand on the old cells (A), and then the change of the cells with the new integrand (B):

$$\mathcal{E}' - \mathcal{E} = \underbrace{\sum_k \int_{V_k} (g_k' - g_k) \, P \, d\xi}_{A} + \underbrace{\sum_k \Big( \int_{V_k'} - \int_{V_k} \Big) g_k' \, P \, d\xi}_{B}.$$

Adding the two parts, the terms $\int_{V_k} g_k' P \, d\xi$ cancel, and we are back to $\mathcal{E}' - \mathcal{E}$.

**The integrand (A).** Only the term $k = c$ changes. Expanding the square, $g_c' - g_c = -2 \, \delta \cdot (\xi - w_c) + \lVert \delta \rVert^2$, so

$$A = -2 \, \delta \cdot \int_{V_c} (\xi - w_c) \, P(\xi) \, d\xi + \lVert \delta \rVert^2 \, P(V_c).$$

The first term is proportional to $\delta$: it is the derivative we are looking for. The second one is of order $\delta^2$.

**The cells (B).** The cells change only in thin strips along the borders between $c$ and its neighbours. Consider the strip $S_{c'}$ along the border with $c'$. A signal in it changes owner, from $c'$ to $c$ or the other way round. Let

$$h(\xi) = g_c'(\xi) - g_{c'}(\xi) = \lVert \xi - w_c - \delta \rVert^2 - \lVert \xi - w_{c'} \rVert^2.$$

A signal that passes from $c'$ to $c$ is now counted with $g_c'$ instead of $g_{c'}$, which changes its contribution by $h(\xi)$, and $h(\xi) \leq 0$ since $c$ is now the closer unit. A signal that passes from $c$ to $c'$ changes its contribution by $-h(\xi) \leq 0$ for the same reason. Therefore

$$B = -\sum_{c'} \int_{S_{c'}} |h(\xi)| \, P(\xi) \, d\xi.$$

Now estimate the size of this term.

- The function $h$ is affine in $\xi$, and it vanishes on the new border. On the old border the two units were equally close, $g_c = g_{c'}$, so there $h = g_c' - g_c = -2 \, \delta \cdot (\xi - w_c) + \lVert \delta \rVert^2$, which is of order $\delta$. Therefore $\lvert h \rvert$ is of order $\delta$ in the whole strip.
- The strip has a width of order $\delta$, since a perpendicular bisector moves by a distance proportional to $\delta$ when one of its two units moves by $\delta$. For a bounded density, its probability is also of order $\delta$.

So $\lvert B \rvert$ is of order $\delta \times \delta = \delta^2$: the shift of the cells costs nothing at first order.

**Result.** Putting A and B together,

$$\mathcal{E}' - \mathcal{E} = -2 \, \delta \cdot \int_{V_c} (\xi - w_c) \, P(\xi) \, d\xi + O(\lVert \delta \rVert^2).$$

The derivative is the coefficient of $\delta$, and the terms of order $\delta^2$ vanish in the limit $\delta \to 0$:

$$\frac{\partial \mathcal{E}}{\partial w_c} = -2 \int_{V_c} (\xi - w_c) \, P(\xi) \, d\xi.$$

So the winner update is a stochastic gradient step on $\mathcal{E}$:

$$\mathbb{E}[\Delta w_c]_{\text{winner}} = -\frac{\varepsilon_b}{2} \, \frac{\partial \mathcal{E}}{\partial w_c} = \varepsilon_b \, P(V_c) \, (m_c - w_c),$$

where $m_c$ is the centroid of the cell,

$$m_c = \frac{1}{P(V_c)} \int_{V_c} \xi \, P(\xi) \, d\xi.$$

>The mean drift of the winner term pulls every unit towards the centroid $m_c$ of its own cell. Its stationary points are the centroidal Voronoi configurations $w_c = m_c$.

>The neighbour term is a second, much weaker pull (since $\varepsilon_n / \varepsilon_b = 0.03$) towards the signals landing in the cells of the neighbours. It couples units along the graph and smooths the configuration. The paper gives no analysis of it, and neither do I.

##### Step 4: the error variable measures the local error

Freeze the units for a moment. The increment of $E_c$ at one signal is

$$X_c = \begin{cases} \lVert \xi - w_c \rVert^2 & \text{if } \xi \in V_c, \\ 0 & \text{otherwise,} \end{cases}$$

so averaging over $\xi$ gives

$$\mathbb{E}[X_c] = D_c, \qquad \sum_c \mathbb{E}[X_c] = \mathcal{E}.$$

The error variable is a running estimate of $D_c$: the unit with the largest error is, on average, the one contributing the most to the quantization error. This is why $q = \arg\max_c E_c$ is where a new unit is needed.

##### Step 9: the decay sets a memory

Let $e_t = \mathbb{E}[E_c]$ at step $t$, for frozen units and without insertions. In the paper's order (add in step 4, decay in step 9), $E_c \leftarrow d(E_c + X_c)$. Taking the expectation, and using $\mathbb{E}[X_c] = D_c$, gives the recursion

$$e_{t+1} = d \, (e_t + D_c).$$

It is a linear first-order difference equation, and it is solved in two steps.

**The fixed point.** Look for a value $e^\ast$ that does not change from one step to the next, $e^\ast = d(e^\ast + D_c)$. Collecting the terms in $e^\ast$, $e^\ast (1 - d) = d \, D_c$, so

$$e^\ast = \frac{d}{1-d} \, D_c.$$

**The distance from the fixed point.** Define $u_t = e_t - e^\ast$. Subtract the equation of the fixed point from the recursion:

$$u_{t+1} = e_{t+1} - e^\ast = d(e_t + D_c) - d(e^\ast + D_c) = d \, (e_t - e^\ast) = d \, u_t.$$

The constant $D_c$ cancels, and what is left is a geometric sequence: $u_t = d^{\,t} u_0$. Going back to $e_t$,

$$e_t = e^\ast + d^{\,t} (e_0 - e^\ast).$$

With $d = 0.995$ this is $e^\ast = 199 \, D_c$, reached with time constant $1/(1-d) = 200$ steps, which is $2\lambda$. So the error variable forgets signals older than about two insertion periods. Without the decay, $e_t = e_0 + t D_c$ grows linearly and keeps the whole history.

##### Step 6 and 7: edges as a statistical test

Freeze the units again, and fix an edge $\lbrace c, c' \rbrace$. Call *endpoint-win* a signal with $s_1 \in \lbrace c, c' \rbrace$, and *refresh* a signal with $\lbrace s_1, s_2 \rbrace = \lbrace c, c' \rbrace$. The age of the edge is incremented at every endpoint-win (step 3) and reset at every refresh (step 6), so it counts the endpoint-wins since the last refresh (or since its creation). The edge is removed when the age reaches $a_{\max} + 1$ (step 7).

An endpoint-win happens with probability $P(V_c) + P(V_{c'})$, of which a fraction

$$\rho = \frac{\rho_{cc'}}{P(V_c) + P(V_{c'})}$$

are refreshes. The signals are iid, so the endpoint-wins are iid trials with success probability $\rho$, and the edge is removed before the next refresh with probability

$$\Pr[\text{removal}] = (1 - \rho)^{a_{\max} + 1} \leq e^{-(a_{\max}+1)\rho}.$$

Two consequences follow.

- An edge outside the induced Delaunay triangulation has $\rho = 0$, so it is removed with certainty, after exactly $a_{\max} + 1$ endpoint-wins.
- An edge of the induced Delaunay triangulation survives with high probability, unless $\rho$ is very small. For $a_{\max} = 50$ and $\rho = 0.1$ the removal probability is $0.9^{51} \approx 4.6 \times 10^{-3}$, and for $\rho = 0.05$ it is $0.95^{51} \approx 7 \times 10^{-2}$. Even if it is removed, step 6 recreates it at the next refresh.

This is why the network tracks the induced Delaunay triangulation, which is a moving target since the units move. The argument is exact only for frozen units. In the real algorithm the units keep moving (with $\varepsilon_b = 0.2$ the winner covers a fifth of the distance to the signal at each win), so $\rho$ changes in time and the bound is a heuristic, not a theorem.

##### Step 8: growth

Each insertion adds one unit and, replacing one edge by two, one net edge. If no unit is ever removed, the number of units after $t$ signals is

$$N(t) = 2 + \left\lfloor \frac{t}{\lambda} \right\rfloor,$$

and in general $N(t) = 2 + \lfloor t/\lambda \rfloor - R(t)$, where $R(t)$ counts the units removed in step 7. Replacing $\lbrace q, f \rbrace$ by a path $q - r - f$ keeps $q$ and $f$ connected, so insertions never disconnect the graph.

The bookkeeping of the errors in step 8 changes their sum by

$$\Delta \sum_c E_c = (2\alpha - 1) E_q - (1 - \alpha) E_f.$$

For $\alpha = \tfrac{1}{2}$ the error of $q$ is conserved, split equally between $q$ and $r$, while $f$ gives away half of its error. This is a heuristic reading: the new unit $r$ takes over part of the territory of both $q$ and $f$, and the errors are reset to what each should plausibly collect afterwards.

The cost of one iteration is dominated by the search of $s_1, s_2$, which is $O(N n)$. Since $N \leq 2 + t/\lambda$, a run of $T$ signals costs $O(T n + T^2 n / \lambda)$.

##### What it looks like

The implementation is in [GrowingNeuralGas](https://github.com/pretidav/GrowingNeuralGas). The data are a cloud of $100000$ points (a subset is drawn in grey), the units are the orange squares, the edges are the black lines, and the black square is the last signal. The title of each picture reports the number of signals (`STEP`), the number of units (`UNITS`), and `ERR`. By the step-9 computation $\sum_c \mathbb{E}[E_c] \approx \frac{d}{1-d} \, \mathcal{E}$, so $\mathcal{E}$ is estimated by $\frac{1-d}{d} \sum_c E_c$, and `ERR` is this estimate divided by the dimension $n$: the mean squared quantization error per dimension.

First, a ring. The distribution is one-dimensional, so the network must discover a closed chain of units: no triangles, and no gaps. In the last picture it has $132$ units and $132$ edges in a single connected component, which is exactly a closed loop. Note how the first units, in the first picture, sit inside the circle.

<div class="fig-row">
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/circle_500.png" caption="500 signals, 7 units." %}
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/circle_3000.png" caption="3000 signals, 32 units." %}
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/circle_13000.png" caption="13000 signals, 132 units." %}
</div>

The same network on a uniform rectangle $[0,2] \times [0,4]$ has to do something very different. The edges now form a two-dimensional triangulation. This is the point of the paper: the dimension of the structure is not chosen in advance, it is whatever the support of $P$ has.

<div class="fig-row">
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/box2d_500.png" caption="500 signals, 7 units." %}
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/box2d_5000.png" caption="5000 signals, 52 units." %}
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/box2d_30500.png" caption="30500 signals, 307 units." %}
</div>

The input space does not have to be the plane. On the closed curve $(\cos t, \sin t, \cos 3t)$ the network finds a one-dimensional chain in $\mathbb{R}^3$ (in this run it has $191$ units and $192$ edges: a loop plus one extra edge), and on the paraboloid patch $(x, y, x^2 + y^2)$ with $x, y \in [3, 13]$ it finds a two-dimensional net.

<div class="fig-row">
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/tilted_19000.png" caption="A closed curve in 3D, 19000 signals, 191 units." %}
{% include figure.html src="/assets/img/2026-11-20-growing-neural-gas/hyppar_27500.png" caption="A paraboloid patch, 27500 signals, 277 units." %}
</div>

The unit counts follow the growth law. With $\lambda = 100$ (the value used for all the pictures), $N(t) = 2 + \lfloor t/100 \rfloor$ gives $7, 32, 132$ for the ring, $7, 52, 307$ for the rectangle and $277$ for the paraboloid, exactly as in the titles. The closed curve shows $191$ instead of $192$: one unit was removed in step 7, so $R = 1$.

##### The code

The implementation has one method per step of the paper.

| Step | Paper | Code (`GNG/neuralgas.py`) |
|---|---|---|
| 0 | two random units | `init` (uniform in $[-1,1]^n$, joined by an edge) |
| 1 | draw $\xi \sim P$ | `distribution.sample`, `embed` (zero padding to `gas_d`) |
| 2 | $s_1, s_2$ | `get_nn` |
| 3 | age of the edges of $s_1$ | `increment_age` |
| 4 | add $\lVert \xi - w_{s_1} \rVert^2$ | `increment_error` |
| 5 | move $s_1$ and neighbours | `update_units` |
| 6 | refresh or create $\lbrace s_1, s_2 \rbrace$ | `reset_age` |
| 7 | remove old edges, then isolated units | `remove_units` |
| 8 | insert $r$ between $q$ and $f$ | `insert_unit`, which calls `find_q`, `find_f`, `add_neuron`, `update_edges` |
| 9 | decay with $d$ | `update_errors` |
| 10 | stopping criterion | `max_number`, `min_number`, `error_tolerance` in `evolve` |

The parameters map as $\lambda \to$ `growing_rate`, $\varepsilon_b \to$ `eps_b`, $\varepsilon_n \to$ `eps_n`, $\alpha \to$ `alpha`, $a_{\max} \to$ `a_max` and $d \to$ `beta`. The error threshold is compared with the estimate of $\mathcal{E}/n$ described above (`quantization_error`), so `error_tolerance` is in the units of the data squared.

Some remarks:
>The edge test is local and needs no global bookkeeping, which is why GNG can keep running after the first topology is found. An edge that the data no longer supports dies after $a_{\max} + 1$ endpoint-wins of its ends.

>The inserted unit sits halfway between two units, which in general is *not* on the data manifold: on the ring it is inside the circle, as the first picture shows, and the first updates pull it onto the data. The paper suspects that this interpolation is what allows constant parameters, since a new unit starts close to where it is needed.

>The distributions in `example/distributions.py` draw a finite point cloud once and then resample it uniformly, so $P$ is an empirical measure and not a density. The derivation of the drift still holds, with the integrals replaced by sums over the points. The edge test, however, feels the finiteness: an edge needs $\rho_{cc'} > 0$, that is, at least one data point whose two closest units are $c$ and $c'$. Once the units are denser than the points, some neighbouring pairs have none and their edge dies. On the ring with $1000$ points, three of four runs of $30000$ signals ended with a ring cut in arcs, while with $100000$ points all four runs ended with one closed ring.

>I checked the formulas numerically. The gradient of $\mathcal{E}$ agrees with finite differences to four digits, and the mean drift with a simulation of steps 2 and 5. For frozen units, the time average of $E_c / D_c$ is between $196.7$ and $199.0$, against the predicted $199$. And the probability of removal before the next refresh, measured by simulating steps 3, 6 and 7, is $0.0186, 0.0074, 0.0017$ for $\rho = 0.076, 0.093, 0.117$, against $(1-\rho)^{51} = 0.0178, 0.0069, 0.0017$.

Cheers.

> References
>1. B. Fritzke. *A Growing Neural Gas Network Learns Topologies.* Advances in Neural Information Processing Systems 7 (NIPS 1994), MIT Press, 1995, pp. 625-632. <https://proceedings.neurips.cc/paper/1994/file/d56b9fc4b0f1be8871f5e1c40c0067e7-Paper.pdf>
>2. T. M. Martinetz, K. J. Schulten. *A "neural-gas" network learns topologies.* In Artificial Neural Networks, North-Holland, 1991, pp. 397-402.
>3. T. M. Martinetz, K. J. Schulten. *Topology representing networks.* Neural Networks 7(3):507-522, 1994.
>4. T. M. Martinetz. *Competitive Hebbian learning rule forms perfectly topology preserving maps.* ICANN'93, Springer, 1993, pp. 427-434.
