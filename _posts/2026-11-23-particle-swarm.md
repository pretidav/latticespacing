---
layout: post
title: "Seek, and Ye Shall Find: Training a Network with a Swarm"
category: ml
---
> Backpropagation needs the gradient of the loss. A particle swarm needs only its value: a crowd of candidate weight vectors flies through parameter space, and each one is pulled towards the best place it has found and the best place the whole crowd has found. Kennedy and Eberhart introduced it in 1995 [1], and trained network weights with it in the very same paper. Let me try to formalize it.

##### Setting

A network $f_\theta$ has $D$ parameters $\theta \in \mathbb{R}^D$, and the loss on the training set is a function $L(\theta)$. I use the network of the example, $4 \to 4 \to 4 \to 3$ with sigmoid, sigmoid and softmax, on the Iris data, and the mean squared error between the softmax output and the one-hot label,

$$\begin{aligned}
f_\theta(x) &= \mathrm{softmax}\Big( W_3 \, \sigma\big( W_2 \, \sigma( W_1 x + b_1 ) + b_2 \big) + b_3 \Big), \\
L(\theta) &= \frac{1}{3N} \sum_{n=1}^{N} \big\lVert f_\theta(x_n) - y_n \big\rVert^2,
\end{aligned}$$

where $N = 75$ is the number of training samples (half of the data, stratified), and $D = (16 + 4) + (16 + 4) + (12 + 3) = 55$. The only thing the optimizer is allowed to do with $L$ is to evaluate it.

##### The algorithm

The swarm has $n$ particles. Particle $i$ has a position $x_i \in \mathbb{R}^D$ (a complete set of weights), a velocity $v_i \in \mathbb{R}^D$, and remembers its best position so far $p_i$. The swarm remembers the best position of all, $g$. The parameters are $w, c_1, c_2 > 0$. One iteration, at time $t$, is

$$v_i^{t+1} = w \, v_i^t + c_1 \, r_1 \odot \big( p_i^t - x_i^t \big) + c_2 \, r_2 \odot \big( g^t - x_i^t \big), \qquad x_i^{t+1} = x_i^t + v_i^{t+1},$$

where $r_1, r_2 \in [0,1]^D$ are vectors of independent uniform random numbers, drawn anew for every particle and every iteration, and $\odot$ is the product of components. After all the particles have moved, the best positions are updated,

$$p_i^{t+1} = \begin{cases} x_i^{t+1} & \text{if } L(x_i^{t+1}) < L(p_i^t), \\ p_i^t & \text{otherwise,} \end{cases} \qquad g^{t+1} = \arg\min_i L(p_i^{t+1}).$$

The first particle starts from the weights of the given network, the others from new random initializations of the same network, and the velocities start uniform in $[-0.1, 0.1]$. An iteration costs $n$ evaluations of $L$ on the whole training set, and no backward pass. Since $g$ is the best of the personal bests and these never get worse, $L(g^t)$ never increases.

The original paper has $w = 1$ and $c_1 = c_2 = 2$. The weight $w$ on the previous velocity is due to Shi and Eberhart [2]. The values I use, $w = 0.7298$ and $c_1 = c_2 = 1.49618$, are the constriction coefficient of Clerc and Kennedy [3]: their update is $v \leftarrow \chi \,( v + \varphi_1 r_1 \odot (p - x) + \varphi_2 r_2 \odot (g - x))$, with $\varphi = \varphi_1 + \varphi_2 > 4$ and

$$\chi = \frac{2}{\varphi - 2 + \sqrt{\varphi^2 - 4 \varphi}}.$$

For $\varphi_1 = \varphi_2 = 2.05$ one has $\varphi = 4.1$ and $\chi = 2 / (2.1 + 0.6403) = 0.7298$, which is the same as the update above with $w = \chi$ and $c_i = \chi \varphi_i = 1.49618$. This is also the choice of the standard PSO of Bratton and Kennedy [4].

##### One particle: why $w$ must be below one

Take a particle that has stopped looking for better places, $p = g = a$ fixed, and shift the origin to $a$. Coordinates are independent, because the random numbers are per component, so it is enough to study one coordinate, with $c_1 = c_2 = c$:

$$v' = w v - \varphi \, x, \qquad x' = x + v', \qquad \varphi = c \,( r_1 + r_2 ),$$

Substituting $v'$ in the second equation gives $x' = (1 - \varphi) x + w v$, so the step is linear in the pair $(x, v)$:

$$\begin{pmatrix} x' \\ v' \end{pmatrix} = A \begin{pmatrix} x \\ v \end{pmatrix}, \qquad A = \begin{pmatrix} 1 - \varphi & w \\ -\varphi & w \end{pmatrix}, \qquad \det A = (1 - \varphi) w + \varphi w = w.$$

Row by row, $A$ acts as

$$\begin{aligned} x' &= (1 - \varphi)\, x + w\, v, \\ v' &= -\varphi\, x + w\, v, \end{aligned}$$

and $\varphi = c\,(r_1 + r_2)$ is drawn anew at every iteration, so $A$ is a random matrix and after $t$ iterations the state is $A_t \cdots A_1 (x_0, v_0)^\top$.

The determinant does not depend on $\varphi$: whatever the random numbers are, one step multiplies areas in the $(x, v)$ plane by $w$. If the particle converged to $a$ whatever its initial condition, the product of the matrices $P_t = A_t \cdots A_1$ would tend to zero, and then $\det P_t = w^t$ would tend to zero too, which needs $w < 1$. For $w = 1$ the product has determinant $1$ and cannot tend to zero: *for every sequence of random numbers there are initial conditions that do not converge.*

For a fixed value of $\varphi$ the eigenvalues of $A$ are the roots of $\lambda^2 - (1 + w - \varphi) \lambda + w = 0$, and both are inside the unit circle if and only if $\lvert w \rvert < 1$ and $0 < \varphi < 2 (1 + w)$. But $\varphi$ is not fixed: it changes at every iteration, and the stability of every single $A$ does not say that the product $A_t \cdots A_1$ goes to zero. For a random $\varphi$ one has to ask how the *mean square* $\mathbb{E}[x_t^2]$ behaves, and I come back to it below, with the map of the parameters.

##### One random number per weight

The random factors must have one entry per weight, as in the original paper. If they were a single number per particle, the new velocity $w v + a (p - x) + b (g - x)$ would be a combination of three vectors, for any $a, b$. With $p$ and $g$ fixed, the three vectors $v$, $p - x$ and $g - x$ span a subspace $S$, the new velocity lies in $S$, and the new $p - x'$ and $g - x'$ are in $S$ too. So the particle stays in the affine subspace $x_0 + S_0$ of dimension at most $3$, in a space of dimension $55$, until $p$ or $g$ change. With one random number per weight the products act as different diagonal matrices on every coordinate, and the steps are not confined.

I tested it, with the switch `per_weight` of the code, on 20 seeds (everything else equal, 30 particles, 200 iterations). Median and 10-90% range:

| | best training loss | test accuracy | spread of the swarm at the end |
|---|---|---|---|
| one random number per weight | 0.0088 [0.0021, 0.0107] | 0.947 [0.944, 0.975] | 2.70 [1.48, 7.08] |
| one random number per particle | 0.0728 [0.0324, 0.1111] | 0.800 [0.667, 0.935] | 0.11 [0.013, 1.70] |

The two ranges of the training loss do not overlap, and the median loss is 8 times larger with a scalar. The swarm also collapses: its spread at the end is 25 times smaller. This is what the subspace argument suggests, but I have not measured the mechanism, only the outcome.

##### Against backpropagation

The reference is Adam, with learning rate $0.01$, $200$ epochs and batches of $8$. The hyperparameters of Adam are my choice, and I tuned neither method. Both are run for 20 seeds, median and 10-90% range:

| | train loss | train accuracy | test loss | test accuracy |
|---|---|---|---|---|
| swarm | 0.0088 [0.0021, 0.0107] | 0.987 [0.987, 1.000] | 0.0261 [0.0126, 0.0367] | 0.947 [0.944, 0.975] |
| Adam | 0.0087 [0.0077, 0.0102] | 0.987 [0.985, 0.987] | 0.0179 [0.0159, 0.0220] | 0.960 [0.959, 0.960] |

{% include figure.html src="/assets/img/2026-11-23-particle-swarm/statistics_iris.png" caption="Left: best training loss of the swarm against the iteration, median and 10-90% band of 20 seeds. Right: the swarm and Adam at equal training time (best loss so far), 20 seeds each." %}

The median training losses are the same, $0.0088$ and $0.0087$. The swarm has a much wider spread: its best seeds go far below Adam, and its worst do not reach it (within its run, 14 of the 20 swarms reach a training loss of $0.01$, against 20 of 20 for Adam, in a median time of $0.4$ s). Its test loss is worse. One test sample is $0.013$ of accuracy, so the accuracies do not separate the methods.

> About the time. A run takes about $0.5$ s for both, on one CPU process. The network has $55$ weights, so this is dominated by the overhead of Python, and it says nothing about larger networks, where the swarm must find a direction in a space of much higher dimension. The comparison is one small data set, one network and untuned Adam.

##### The health of the swarm

The best loss is a single number. To see what the swarm is doing I record, at every iteration, the spread of the particles around their centroid (rms over particles and weights), the rms velocity, the minimum, median and maximum current loss, and the fraction of particles that improve their best position. The figure compares the default $w = 0.73$ with $w = 1$, with the same $c_1 = c_2$.

{% include figure.html src="/assets/img/2026-11-23-particle-swarm/health_iris.png" caption="Diversity, velocity, current losses of the particles and fraction of particles that improve (10-iteration mean). Orange: w = 0.73, black: w = 1. Seed 0." %}

With $w = 1$ the spread and the velocity grow exponentially, from about $0.1$ to about $10^7$ in 200 iterations, the best loss stops at about $0.1$, and after iteration 75 no particle improves. The determinant argument says that there is no contraction at $w = 1$. That the particles actually fly apart, exponentially, is what I observe, and I did not derive it (a unit determinant combined with a random shear stretches one direction as much as it compresses the other). With the default the spread grows to about $5$ in the first $40$ iterations and then decays slowly, to about $1.5$ at the end: the swarm has not contracted yet in 200 iterations, and $5$ to $15\%$ of the particles still improve at every step. It has not converged prematurely, and it is not finished.

##### The map of the parameters

For a stagnating particle, with $\varphi = c\,(r_1 + r_2)$ random, the mean is not the right quantity: $\mathbb{E}[x_t]$ can go to zero while the particle swings between large positive and negative values. The mean square $\mathbb{E}[x_t^2]$ detects this, and it obeys an exact linear recursion. Squaring the two rows of the step gives $x'^2$, $x'v'$ and $v'^2$ as combinations of $x^2$, $xv$ and $v^2$, with coefficients that depend on $\varphi$. Since $\varphi$ is drawn independently of $(x, v)$, taking the expectation replaces the coefficients by their means, which need only $\mathbb{E}[\varphi] = c$ and $\mathbb{E}[\varphi^2] = \tfrac{7}{6} c^2$ (the sum $r_1 + r_2$ has mean $1$ and variance $\tfrac{1}{6}$). The three expansions are

$$\begin{aligned} x'^2 &= (1 - \varphi)^2\, x^2 + 2 w (1 - \varphi)\, x v + w^2\, v^2, \\ x' v' &= (\varphi^2 - \varphi)\, x^2 + w (1 - 2 \varphi)\, x v + w^2\, v^2, \\ v'^2 &= \varphi^2\, x^2 - 2 w \varphi\, x v + w^2\, v^2, \end{aligned}$$

and their means, with $\mathbb{E}[(1 - \varphi)^2] = 1 - 2c + \tfrac{7}{6} c^2$, $\mathbb{E}[1 - \varphi] = 1 - c$, $\mathbb{E}[\varphi^2 - \varphi] = \tfrac{7}{6} c^2 - c$, $\mathbb{E}[1 - 2\varphi] = 1 - 2c$, are the rows of a $3 \times 3$ matrix. The vector $m_t = (\mathbb{E} x_t^2, \mathbb{E} x_t v_t, \mathbb{E} v_t^2)^\top$ obeys $m_{t+1} = M m_t$, that is

$$\begin{pmatrix} \mathbb{E} x_{t+1}^2 \\ \mathbb{E} x_{t+1} v_{t+1} \\ \mathbb{E} v_{t+1}^2 \end{pmatrix} = \begin{pmatrix} 1 - 2c + \tfrac{7}{6} c^2 & 2 w (1 - c) & w^2 \\ \tfrac{7}{6} c^2 - c & w (1 - 2c) & w^2 \\ \tfrac{7}{6} c^2 & -2 w c & w^2 \end{pmatrix} \begin{pmatrix} \mathbb{E} x_t^2 \\ \mathbb{E} x_t v_t \\ \mathbb{E} v_t^2 \end{pmatrix},$$

and the matrix $M$ depends only on $c$ and $w$ (it is in `plots.stagnation_rate`). So $m_t = M^t m_0$, and the particle contracts in mean square if and only if $\rho(M) < 1$, where $\rho(M)$ is the spectral radius, the largest modulus of the eigenvalues of $M$. The mean square changes by a factor $\rho(M)$ per step, which is a rate $\tfrac{1}{2} \log \rho(M)$ for the size of $x$. The boundary $\rho(M) = 1$ is the condition of Poli [5],

$$c_1 + c_2 < \frac{24 \, (1 - w^2)}{7 - 5 w}.$$

The left of the figure is the rate $\tfrac{1}{2} \log \rho(M)$ over the whole plane $(c, w)$, with $c_1 = c_2 = c$. On the right of the figure is the best loss on Iris for the same grid (100 iterations, 15 particles, 2 seeds per cell).

{% include figure.html src="/assets/img/2026-11-23-particle-swarm/stability_map_iris.png" caption="Left: growth rate of the second moment of a stagnating particle; the contour of zero and the dashed line are the boundary of Poli. Right: log10 of the best training loss on Iris (light: lower). The cross is the default." %}

The loss is high on both sides: outside the boundary the swarm spreads, and deep inside it contracts too early. The cells with the lowest loss form a band just inside the boundary. The default is in the neighbourhood of that band, but a few cells are better, for example $(c, w) = (1.0, 0.8)$. With two seeds and a coarse grid I would not read the individual cells. What the picture supports is the shape: the useful parameters are those that are stable, but not by a large margin.

##### Generalization

The swarm only minimizes the training loss, and it does not keep the weights small. I recorded the training and test loss of the best model, and its largest weight, at every iteration, for five seeds.

{% include figure.html src="/assets/img/2026-11-23-particle-swarm/generalization_iris.png" caption="Training loss, test loss and largest weight of the best model against the iteration, 5 seeds. Orange: seed 1." %}

| seed | final train loss | final test loss | lowest test loss (iteration) | largest weight at the end |
|---|---|---|---|---|
| 0 | 0.0063 | 0.0282 | 0.0097 (111) | 44 |
| 1 | 0.00003 | 0.0354 | 0.0279 (92) | 57 |
| 2 | 0.0079 | 0.0217 | 0.0131 (110) | 113 |
| 3 | 0.0088 | 0.0266 | 0.0187 (80) | 100 |
| 4 | 0.0106 | 0.0289 | 0.0104 (75) | 16 |

The training loss keeps decreasing, but in all five seeds the test loss has its minimum between iterations 75 and 111 and ends 1.3 to 2.9 times higher, while the largest weight grows to between 16 and 113. Seed 1 fits the training set exactly (accuracy 1) and has the worst test loss. A saturated softmax needs large weights, and nothing stops the swarm from finding them.

> The lowest test loss in the table is picked on the test set, so it is optimistic: it shows that there is a better moment to stop, not how much one would gain. A stopping rule needs a validation set taken out of the training data. With five seeds I cannot say whether the size of the weights explains the differences between seeds.

##### The implementation

The code is in `pso/`: `pso.py` has the swarm, and the diagnostics are recorded by it.

| Object | Code |
|---|---|
| $x_i$, $v_i$, $p_i$, $g$ | `Swarm.X`, `Swarm.V`, `Swarm.P`, `Swarm.G` (one row per particle) |
| the loss $L(\theta)$ | `Swarm._fitness` (the weights are copied into the network, no gradient) |
| one iteration | `Swarm.fit` |
| $r_1$, $r_2$ with one number per weight | `torch.rand(n, D)`, switch `per_weight` |
| spread, velocity, losses, improving fraction | `Swarm.stats`, and `callback` in `fit` |
| $\tfrac{1}{2}\log \rho(M)$ | `plots.stagnation_rate` |

The parameters are `w`, `c1`, `c2`, `n_particles`, and the initial velocity `init_velocity`. A `seed` fixes every random number of the swarm without touching the global generator of PyTorch. Eight tests compare one iteration with the update computed directly from the same random numbers, check that the factors differ for every weight (and are equal with the switch off), that the best loss never increases, that the given model is not modified, that the diagnostics have the right lengths, and that runs are reproducible.

Some remarks:
>The update is synchronous: all the particles move with the same $p_i$ and $g$, and then the best positions are updated. Updating $g$ after every particle is also legitimate, but then the result depends on the order of the particles.

>The topology is global: every particle follows the best of the whole swarm. Other neighbourhoods exist (a ring is a common one) and I did not try them.

>The loss is always computed on the whole training set. A mini-batch would make the loss of a particle and its best loss refer to different functions, and the comparison $L(x) < L(p)$ would be partly a lucky batch. I did not measure how much this matters here.

Cheers.

> References
>1. J. Kennedy, R. Eberhart. *Particle swarm optimization.* Proceedings of the IEEE International Conference on Neural Networks, vol. 4, pp. 1942-1948, 1995.
>2. Y. Shi, R. Eberhart. *A modified particle swarm optimizer.* Proceedings of the IEEE International Conference on Evolutionary Computation, pp. 69-73, 1998.
>3. M. Clerc, J. Kennedy. *The particle swarm: explosion, stability, and convergence in a multidimensional complex space.* IEEE Transactions on Evolutionary Computation 6(1):58-73, 2002.
>4. D. Bratton, J. Kennedy. *Defining a standard for particle swarm optimization.* IEEE Swarm Intelligence Symposium, pp. 120-127, 2007.
>5. R. Poli, J. Kennedy, T. Blackwell. *Particle swarm optimization: an overview.* Swarm Intelligence 1:33-57, 2007.
