---
layout: post
title: "Dare to Remember: Memory at Finite Temperature"
category: phys
---
> A Hopfield network stores a few patterns as the valleys of an energy, and recalls one of them by rolling downhill from a corrupted copy. Real neurons are noisy, and noise is a temperature: when it is small the valleys are still there, and when it is large the network forgets everything. Hopfield introduced the model in 1982 [1], and Amit, Gutfreund and Sompolinsky studied it as a spin glass at finite temperature [3, 4]. Let me try to formalize it.

##### Setting

There are $N$ units $S_i \in \lbrace +1, -1 \rbrace$ and $M$ patterns $\xi^\mu \in \lbrace +1, -1 \rbrace^N$, $\mu = 1, \dots, M$, whose entries are independent fair coins. The weights are the Hebbian ones, without self-coupling, and the field on unit $i$ is $h_i$:

$$W_{ij} = \frac{1}{N} \sum_{\mu=1}^{M} \xi_i^\mu \xi_j^\mu \quad (i \neq j), \qquad W_{ii} = 0, \qquad h_i = \sum_j W_{ij} S_j.$$

The matrix is symmetric, and this is what makes everything below work. The energy of a state and the overlap of a state with the pattern $\mu$ are

$$E(S) = -\frac{1}{2} \sum_{i,j} W_{ij} S_i S_j, \qquad m_\mu = \frac{1}{N} \sum_i \xi_i^\mu S_i.$$

> The energy is reminiscent of the Ising model, $-J \sum_{\langle ij \rangle} S_i S_j$, but the interaction is not restricted to nearest neighbours: every unit is coupled to every other one, with couplings $W_{ij}$ of either sign that are set by the patterns.

The energy can be written with the overlaps alone. Since $W_{ii} = 0$, only the pairs $i \neq j$ appear, and for those I insert the definition of $W_{ij}$ and exchange the order of the sums:

$$E(S) = -\frac{1}{2N} \sum_{\mu=1}^{M} \sum_{i \neq j} \big( \xi_i^\mu S_i \big) \big( \xi_j^\mu S_j \big).$$

The sum over the pairs $i \neq j$ is the sum over all pairs, which is a square, minus the terms with $i = j$:

$$\sum_{i \neq j} \big( \xi_i^\mu S_i \big) \big( \xi_j^\mu S_j \big) = \Big( \sum_i \xi_i^\mu S_i \Big)^2 - \sum_i \big( \xi_i^\mu S_i \big)^2 = N^2 m_\mu^2 - N.$$

In the last step the square is $(N m_\mu)^2$ by the definition of the overlap, and each of the $N$ terms with $i = j$ is equal to $1$, because $\xi_i^\mu$ and $S_i$ are $\pm 1$. Inserting this in the energy and dividing by $N$,

$$\frac{E}{N} = -\frac{1}{2} \sum_{\mu=1}^{M} m_\mu^2 + \frac{M}{2N}.$$

The energy is low when the overlap with a stored pattern is large in absolute value, whether it is positive or negative. In particular $E(-S) = E(S)$, so whenever the network stores a pattern $\xi^\mu$, it also stores the reversed pattern $-\xi^\mu$.

> The field too can be written with the overlaps. Inserting the weights,
> $$h_i = \frac{1}{N} \sum_{\mu=1}^{M} \xi_i^\mu \sum_{j \neq i} \xi_j^\mu S_j = \sum_{\mu=1}^{M} \xi_i^\mu m_\mu - \frac{M}{N} S_i,$$
> because the sum over $j \neq i$ is $N m_\mu - \xi_i^\mu S_i$, and $(\xi_i^\mu)^2 = 1$. For $M/N \to 0$ the last term is negligible: the field on a unit is the sum of its values in the patterns, each weighted by the overlap of the state with that pattern. The overlaps create the field, and the field decides what the units do.

It remains to check that the network keeps a stored pattern. Without noise a unit takes the sign of its field, so a state is left unchanged, that is, it is a fixed point, if $S_i = \mathrm{sign}(h_i)$ for every unit $i$. Put the network in the pattern $\nu$, $S = \xi^\nu$, and call $h_i^\nu$ the field in this state:

$$h_i^\nu = \sum_j W_{ij} \xi_j^\nu = \frac{1}{N} \sum_{\mu=1}^{M} \sum_{j \neq i} \xi_i^\mu \xi_j^\mu \xi_j^\nu.$$

In the term $\mu = \nu$ of the sum $\xi_j^\nu \xi_j^\nu = 1$, so that term is $\frac{N-1}{N} \xi_i^\nu$. Separating it from the others defines $\gamma_i^\nu$:

$$h_i^\nu = \frac{N-1}{N} \, \xi_i^\nu + \gamma_i^\nu, \qquad \gamma_i^\nu = \frac{1}{N} \sum_{\mu \neq \nu} \sum_{j \neq i} \xi_i^\mu \xi_j^\mu \xi_j^\nu.$$

The first term is the signal, which has the sign of the pattern. The second one, the crosstalk, is the disturbance due to the other patterns: it is a sum of $(M-1)(N-1)$ independent terms $\pm 1$, divided by $N$, so it has mean $0$ and variance $(M-1)(N-1)/N^2 \approx M/N$.

> I take $M$ fixed and $N$ large, so that $M/N \to 0$. The crosstalk is then small compared with the signal: by Hoeffding's inequality $\lvert \gamma_i^\nu \rvert \geq 1/2$ with probability at most $2 e^{-N/(8M)}$, and this stays small after summing over the $N$ units. So with high probability $\mathrm{sign}(h_i^\nu) = \xi_i^\nu$ for all $i$ at once, and every stored pattern is a fixed point [2].

##### Noisy units

At temperature $T = 1/\beta$ a unit that is updated takes the value $\pm 1$ with probability

$$P(S_i = \pm 1 \mid h_i) = \frac{1}{2} \Big( 1 \pm \tanh(\beta h_i) \Big) = \frac{1}{1 + e^{\mp 2 \beta h_i}},$$

and a sweep updates the $N$ units one at a time, in a random order, each with the current value of the others. This is the heat-bath (Glauber) dynamics of the Ising model [6]. The limit $T \to 0$ is the deterministic rule, with $\mathrm{sign}(0) = +1$, and $T \to \infty$ is a fair coin.

> The reason for this rule is that it samples the Gibbs distribution $p(S) = e^{-\beta E(S)} / Z$. The terms of $-\beta E$ that contain $S_i$ are $\frac{\beta}{2} (W_{ij} S_i S_j + W_{ji} S_j S_i)$ summed over $j \neq i$, which by symmetry is $\beta S_i h_i$, with $h_i$ not containing $S_i$ because $W_{ii} = 0$. Hence
> $$p(S_i = \pm 1 \mid S_{-i}) = \frac{e^{\pm \beta h_i}}{e^{\beta h_i} + e^{-\beta h_i}} = \frac{1}{1 + e^{\mp 2 \beta h_i}},$$
> which is the rule. The same fact says that the update satisfies detailed balance. Flipping the unit $i$ changes the energy by $\Delta E = 2 S_i h_i$ (the term $-S_i h_i$ changes sign), and the probability of the flip is $1 / (1 + e^{\beta \Delta E})$, so for the flip $S \to \tilde S$ and its reverse
>$$\frac{P(S \to \tilde S)}{P(\tilde S \to S)} = \frac{1 + e^{-\beta \Delta E}}{1 + e^{\beta \Delta E}} = e^{-\beta \Delta E} = \frac{p(\tilde S)}{p(S)}.$$

At $T = 0$ a flip happens only if $\Delta E \leq 0$, and then the energy cannot increase and the dynamics stops in a local minimum. At $T > 0$ a flip that raises the energy by $\Delta E$ is allowed with probability $1/(1 + e^{\beta \Delta E})$, and the energy fluctuates around an average value instead of decreasing.

##### Mean field

The mean field approximation, which I follow from Orhan's note [5], writes $S_i = s_i + \delta S_i$, where $s_i = \langle S_i \rangle$ is the mean and $\delta S_i$ is the fluctuation. In the Gibbs weight

$$\frac{\beta}{2} \sum_{i,j} W_{ij} (s_i + \delta S_i)(s_j + \delta S_j) = \frac{\beta}{2} \sum_{i,j} W_{ij} ( s_i s_j + s_i \delta S_j + s_j \delta S_i ) + \mathcal{O}(\delta S_i \delta S_j),$$

put back $\delta S_i = S_i - s_i$. The bracket becomes

$$s_i s_j + s_i (S_j - s_j) + s_j (S_i - s_i) = s_i S_j + s_j S_i - s_i s_j.$$

The two terms that contain a unit give the same sum, because $W_{ij} = W_{ji}$ and the names of the indices can be exchanged, and the last term does not contain any unit. Therefore

$$\begin{aligned}
\frac{\beta}{2} \sum_{i,j} W_{ij} \big( s_i S_j + s_j S_i - s_i s_j \big) &= \beta \sum_{i,j} W_{ij} s_i S_j - \frac{\beta}{2} \sum_{i,j} W_{ij} s_i s_j \\
&= \beta \sum_j S_j \Big( \sum_i W_{ij} s_i \Big) + \mathrm{const}.
\end{aligned}$$

The term in brackets is the average of the field, $\langle h_j \rangle = \sum_i W_{ji} s_i$. The constant is absorbed in the normalization, and the distribution factorizes into one factor per unit,

$$p(S) \approx \frac{1}{Z^{\prime}} \prod_{j=1}^{N} e^{\beta \langle h_j \rangle S_j}.$$

Each unit is now independent of the others and feels only their mean field. Its average, over the two values $\pm 1$, is

$$s_j = \frac{e^{\beta \langle h_j \rangle} - e^{-\beta \langle h_j \rangle}}{e^{\beta \langle h_j \rangle} + e^{-\beta \langle h_j \rangle}} = \tanh\Big( \beta \sum_i W_{ji} s_i \Big).$$

Look for a state that is a multiple of one pattern, $s_i = m \, \xi_i^\nu$. The field on unit $j$ is then $m$ times the field of the pattern, which was computed in the setting:

$$\sum_i W_{ij} m \, \xi_i^\nu = m \, h_j^\nu = m \Big( \frac{N-1}{N} \, \xi_j^\nu + \gamma_j^\nu \Big).$$

For $M/N \to 0$ I neglect the crosstalk $\gamma_j^\nu$ and the factor $(N-1)/N$. The equation becomes $m \xi_j^\nu = \tanh(\beta m \xi_j^\nu)$, and since $\tanh$ is odd the sign of $\xi_j^\nu$ drops out:

$$m = \tanh(\beta m).$$

> For $M/N \to 0$ I expect this to be exact, because every unit interacts with all the others through couplings of size $1/N$, so that its field is a sum of many small terms.

##### The transition at $T = 1$

Let $g(m) = \tanh(\beta m) - m$ for $m \geq 0$. Then $g(0) = 0$, $g^{\prime}(0) = \beta - 1$, $g^{\prime\prime} < 0$ (because $\tanh$ is concave on the positive axis) and $g(1) < 0$. If $T > 1$, then $\beta < 1$, $g^{\prime}(0) < 0$ and $g$ is concave: $g < 0$ for all $m > 0$, and $m = 0$ is the only solution (equivalently $\tanh(\beta m) \leq \beta m < m$). If $T < 1$, then $g$ starts upwards, is concave and ends negative, so it has exactly one positive zero $m_\star(T)$, with $g^{\prime}(m_\star) < 0$: the map $m \mapsto \tanh(\beta m)$ has slope $\beta (1 - m_\star^2) < 1$ there, so the solution is stable, while $m = 0$, where the slope is $\beta > 1$, is not. For $T \to 1^-$, expanding $\tanh x = x - x^3/3$,

$$m_\star^2 \approx \frac{3 (\beta - 1)}{\beta^3} \quad \Longrightarrow \quad m_\star \approx \sqrt{3 (1 - T)},$$

a continuous transition with exponent $1/2$. Numerically, $m_\star(0.99) = 0.1725$ against $\sqrt{3 \cdot 0.01} = 0.1732$, and $m_\star(0.95) = 0.3795$ against $0.3873$. Below $T_c = 1$ the network keeps a memory, above it the memory is gone.

I simulated it with $N = 1000$ units, $M = 3$ patterns, 10 replicas per temperature and 30 sweeps, starting exactly on the first pattern. The left panel shows the overlap with it, the right panel the energy per unit after the last sweep.

{% include figure.html src="/assets/img/2026-11-25-hopfield/m_E_T_1000-3.png" caption="Left: overlap of the final state with the stored pattern against the temperature, mean and standard error of 10 replicas, against the mean field curve m = tanh(m/T). Right: energy per unit after the last sweep. N = 1000, M = 3, 30 sweeps." %}

The numbers for a few temperatures:

| temperature | overlap, mean field | overlap, simulation | energy per unit, simulation | minus half the squared overlap (simulated) |
|---|---|---|---|---|
| 0.55 | 0.9355 | 0.930 | $-0.432$ | $-0.432$ |
| 0.75 | 0.7755 | 0.771 | $-0.297$ | $-0.297$ |
| 0.85 | 0.6295 | 0.635 | $-0.204$ | $-0.202$ |
| 0.95 | 0.3795 | 0.299 | $-0.069$ | $-0.045$ |
| 1.05 | 0 | 0.015 | $-0.023$ | $-0.0001$ |

Up to $T = 0.85$ the simulated overlap agrees with $m_\star$ within $0.006$, and above $T_c$ it is zero within the errors. Close to $T_c$ it is below the prediction (at $T = 0.95$, $0.299$ against $0.380$): this is where the neglected fluctuations are largest, and with $N = 1000$ and 30 sweeps I do not separate a finite-size effect from a slow relaxation.

##### The energy

With one pattern retrieved the identity of the setting gives $E/N \approx -m^2/2$, plus $M/(2N) = 0.0015$ and the small overlaps of the other patterns. The last two columns of the table test it: the energy per unit is the one of the overlap alone up to $T = 0.85$, to within $0.003$. The deviation at $T = 0.95$ and above $T_c$ is the same effect as before. There the overlaps are not zero but fluctuate, with $\langle m_\mu^2 \rangle$ of order $1/N$ for every pattern, the energy is $-\frac{1}{2} \sum_\mu m_\mu^2$ of those fluctuations, and it stays negative while $m = 0$. I did not measure this quantitatively.

The energy against the sweeps, starting on the pattern at $E/N = -0.5$:

{% include figure.html src="/assets/img/2026-11-25-hopfield/energy_t_1000-3.png" caption="Energy per unit against the number of sweeps for six temperatures, mean and standard error of 10 replicas. N = 1000, M = 3, start on the pattern." %}

At $T = 0.05$ the state sits at the bottom of the valley. For larger $T$ the energy goes up and settles on a plateau, higher as the temperature grows, and for $T > 1$ it goes to $0$. The plateau is not a minimum: the dynamics samples the Gibbs distribution, whose average energy is above the minimum. The time to reach it is short. For $T = 0.55$, $0.75$ and $0.85$, in a run that I did not plot, the mean energy is within $0.01$ of its late value after 1, 4 and 4 sweeps. This is the relaxation inside one valley, which is not the same thing as the time to escape from it.

##### The other edge: the load at zero temperature

Everything above has $M/N \to 0$. When $\alpha = M/N$ is finite the crosstalk is not negligible, and it acts as noise even at $T = 0$. The mean field equations for $m$ and the two order parameters $q$ and $r$ are in the note [5] (and in [3, 4]). At $T = 0$ they reduce to a single equation, with $y = m / \sqrt{2 \alpha r}$,

$$y \Big( \sqrt{2 \alpha} + \frac{2}{\sqrt{\pi}} e^{-y^2} \Big) = \mathrm{erf}(y), \qquad m = \mathrm{erf}(y).$$

The retrieval state is the largest root. It exists up to $\alpha_c$ and disappears there discontinuously: numerically $\alpha_c = 0.1379$, with $m = 0.9674$ at that point. At the other end, $m = 0.998$ at $\alpha = 0.1$ and $0.993$ at $\alpha = 0.12$.

The simulation, with $N = 500$, starting on the pattern and relaxing at $T = 0$:

{% include figure.html src="/assets/img/2026-11-25-hopfield/capacity_500.png" caption="Overlap of the final state with the stored pattern against the load alpha = M/N, at T = 0, N = 500, mean and standard error of 20 sets of patterns. The dashed line is the mean field alpha_c = 0.138." %}

The overlap is above $0.97$ up to $\alpha = 0.12$, and it falls over a range of $\alpha$, to $0.44$ at $0.2$, instead of jumping at $0.138$. The mean field value is the limit $N \to \infty$, so I repeated the scan for $N = 100$ to $3200$ (40 sets of patterns for every point, at most 100 sweeps), and called $\alpha_c(N)$ the load at which half of the runs still end with $m > 0.5$. The fraction of those runs that retrieve:

{% include figure.html src="/assets/img/2026-11-25-hopfield/finite_size_retrieval.png" caption="Fraction of the runs whose final overlap is above 0.5 against the load, for N from 100 to 3200 (T = 0, start on the pattern). The dashed line is the mean field alpha_c." %}

The curves become steeper with $N$ and move towards the dashed line. The load $\alpha_c(N)$ is $0.327$, $0.248$, $0.199$, $0.170$, $0.1635$, $0.156$ for $N = 100, 200, 400, 800, 1600, 3200$, and it is still decreasing at the largest size.

{% include figure.html src="/assets/img/2026-11-25-hopfield/finite_size_capacity.png" caption="The load alpha_c(N) against 1/sqrt(N), with error bars from a bootstrap over the runs, and two fits of a + b N^(-nu): nu = 1/2 fixed (dotted, grey diamond at N = infinity) and nu free (dashed, hollow square). The line is the mean field alpha_c." %}

How it goes to the limit is not known to me, so I fitted $\alpha_c(N) = a + b N^{-\nu}$ with several exponents:

| fit | extrapolated capacity | chi-squared (4 degrees of freedom) |
|---|---|---|
| $\nu = 1/2$ fixed | $0.128 \pm 0.003$ | 15.5 |
| $\nu = 0.67$ fixed | $0.139 \pm 0.002$ | 8.1 |
| $\nu = 1$ fixed | $0.151 \pm 0.002$ | 2.4 |
| $\nu$ free | $0.1505 \pm 0.0016$ at $\nu = 0.98$ | 2.3 |

The data prefer $\nu$ close to $1$ ($\nu$ between $0.86$ and $1.10$ within one unit of $\chi^2$, with $a$ between $0.147$ and $0.153$), and they disfavour $1/\sqrt{N}$. The extrapolated value depends on the exponent that I assume, so I do not read it as a measurement of $0.138$: it is below it for $\nu = 1/2$, and about $0.012$ above it for the best $\nu$. What the scan supports is that $\alpha_c(N)$ decreases towards a number between $0.13$ and $0.15$, which contains the mean field value but does not fix it.

##### The implementation

| object | code |
|---|---|
| $W$, $h_i$, $E$, $m_\mu$ | `HopfieldNetwork.W`, `.energy`, `.overlaps` |
| one update with probability $\frac{1}{2}(1 + \tanh(\beta h_i))$ | `HopfieldNetwork.update_unit` |
| a sweep (random order), a run of sweeps | `.sweep`, `.run` |
| relaxation at $T = 0$ until no unit flips | `.relax` |
| $m = \tanh(m/T)$ | `mean_field_m` |
| the equation for $y$, $\alpha_c$ | `mean_field_retrieval`, `mean_field_capacity` |
| scans over $T$, $\alpha$, $(N, \alpha)$ | `temperature_scan`, `capacity_scan`, `finite_size_scan` |
| $\alpha_c(N)$ and the fit | `alpha_c_of_N`, `fit_extrapolation` |

Some remarks:
>The mean field step that neglects $\delta S_i \delta S_j$ is not controlled in general. For $M/N \to 0$ the simulation agrees with it, and the agreement degrades close to $T_c$ and for $\alpha$ near $\alpha_c$, which is where I would expect the correlations to matter.

>The symmetry of $W$ is essential: without it there is no energy, no Gibbs distribution, and no detailed balance. I used asynchronous updates. With synchronous updates the same rule does not sample $e^{-\beta E}$, and the dynamics can cycle.

>I started every run on a stored pattern, so what I measure is the stability of the memory, not the size of its basin of attraction. Starting from a corrupted pattern, or from a random state, would also show the spurious states, mixtures of an odd number of patterns, which I did not look at.

Cheers.

> References
>1. J. J. Hopfield. *Neural networks and physical systems with emergent collective computational abilities.* Proceedings of the National Academy of Sciences 79(8):2554-2558, 1982.
>2. J. Hertz, A. Krogh, R. G. Palmer. *Introduction to the Theory of Neural Computation.* Addison-Wesley, 1991.
>3. D. J. Amit, H. Gutfreund, H. Sompolinsky. *Spin-glass models of neural networks.* Physical Review A 32:1007-1018, 1985.
>4. D. J. Amit, H. Gutfreund, H. Sompolinsky. *Storing infinite numbers of patterns in a spin-glass model of neural networks.* Physical Review Letters 55:1530-1533, 1985.
>5. E. Orhan. *The Hopfield model.* Lecture note, 2014.
>6. R. J. Glauber. *Time-dependent statistics of the Ising model.* Journal of Mathematical Physics 4:294-307, 1963.
>7. W. Gerstner, W. M. Kistler, R. Naud, L. Paninski. *Neuronal Dynamics.* Cambridge University Press, 2014, chapter 17.
