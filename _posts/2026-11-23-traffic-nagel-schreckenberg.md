---
layout: post
title: "The Last Shall Be First: Jams on a Cellular Road"
category: phys
---
> A road is a ring of $L$ sites, a car is a particle with an integer speed, and the only randomness is that a driver sometimes lifts the foot from the pedal. This is the Nagel-Schreckenberg cellular automaton [1]: four rules, one parameter $p$, and yet it produces free flow, start-stop waves that travel backwards, and a fundamental diagram with a maximum. I re-implemented it, together with its extension to two lanes with asymmetric lane changes [2], and in this post I derive what can be derived, and measure the rest. Let me try to formalize it.

##### Setting

The road is the ring $\mathbb{Z}/L\mathbb{Z}$. There are $N$ cars at distinct sites $x_1 < x_2 < \dots < x_N$ (cyclically), with density $\rho = N/L$. Car $i$ has an integer speed $v_i \in \lbrace 0, 1, \dots, v_{\max} \rbrace$, and the *gap* to the car ahead is the number of empty sites between them,

$$g_i = x_{i+1} - x_i - 1, \qquad \sum_{i=1}^{N} g_i = L - N,$$

where $x_{N+1} = x_1 + L$. One time step applies four rules in parallel to all cars [1]:

- **1. Acceleration:** $v_i \leftarrow \min(v_i + 1, v_{\max})$.
- **2. Slowing down:** $v_i \leftarrow \min(v_i, g_i)$.
- **3. Randomization:** with probability $p$, independently for each car, $v_i \leftarrow \max(v_i - 1, 0)$.
- **4. Motion:** $x_i \leftarrow x_i + v_i$.

Rule 2 is what makes collisions impossible: after it, the car cannot reach the site of the car ahead, and since the car ahead only moves forward, no two cars ever share a site. The paper uses $v_{\max} = 5$ and I take $p = 0.5$ throughout, which is the value for which the numbers of the paper come out.

> My code writes the distance to the car ahead as $d_i = g_i + 1$ and accelerates only if $d_i > v_i + 1$. This is the same rule: if $g_i > v_i$ both give $v_i + 1$, if $g_i = v_i$ both leave $v_i$ (the paper's $v_i + 1$ is cut back to $g_i = v_i$ by rule 2), and if $g_i < v_i$ both end at $g_i$.

##### The flow

The observable is the flow $q$: the average number of cars that cross a bond per time step. In one step car $i$ crosses exactly $v_i$ bonds, so the total number of crossings is $\sum_i v_i$. Averaging over the $L$ bonds of the ring,

$$q = \frac{1}{L} \sum_{i=1}^{N} v_i = \rho \, \langle v \rangle, \qquad \langle v \rangle = \frac{1}{N} \sum_{i=1}^{N} v_i.$$

In a stationary state every bond carries the same average flow, so this is also the flow through any single bond. (My first version of the code averaged a cumulative counter instead, and got half of this.)

> A bound that holds at every time step, for every $p$. After rule 2, $v_i \leq g_i$, and rule 3 can only lower $v_i$. Summing over the cars, $\sum_i v_i \leq \sum_i g_i = L - N$, and $v_i \leq v_{\max}$ gives $\sum_i v_i \leq v_{\max} N$. Dividing by $L$,
> $$q \leq \min\big( v_{\max} \, \rho, \ 1 - \rho \big).$$

##### Free flow

At low density a car never feels the others, and its gap never matters. Then rules 1, 3 and 4 act on the speed alone. Once the speed is in $\lbrace v_{\max} - 1, v_{\max} \rbrace$, it stays there: from either value rule 1 gives $v_{\max}$, and rule 3 lowers it to $v_{\max} - 1$ with probability $p$. So after each step

$$v = v_{\max} - \xi, \qquad \xi \sim \text{Bernoulli}(p) \text{ independent at every step.}$$

The set $\lbrace v_{\max} - 1, v_{\max} \rbrace$ is reached in finite time with probability one starting from any speed, if $p < 1$: below it, the speed increases by one with probability $1 - p$ and otherwise stays where it is. Therefore $\langle v \rangle = v_{\max} - p$ and

$$q = (v_{\max} - p) \, \rho \qquad \text{(free flow).}$$

For $v_{\max} = 5$, $p = 0.5$ this is $4.5 \rho$, and the simulation gives $\langle v \rangle = 4.499$, $4.498$, $4.494$ at $\rho = 0.005, 0.01, 0.02$. The mean speed in the figure sits on this plateau, $4.5 / 5 = 0.9$, up to $\rho \approx 0.06$.

{% include figure.html src="/assets/img/2026-11-23-traffic/fundamental_diagram_L1000_p0.5_vmax5.png" caption="Ring road, L = 1000, p = 0.5, vmax = 5. Flow q (orange) and mean speed over vmax (black) against the density, each point a separate run of 50000 steps after a relaxation of 10 L steps. The dashed line is the bound for p = 0. The cross is the open road of the section below: a different experiment, not a point of the ring." %}

The flow grows linearly up to a maximum $q \approx 0.33$ at $\rho \approx 0.08$, and then decreases: this is the change-over between free flow and jammed flow. With $p > 0$ it is a crossover whose position and shape depend on the size of the system [1], not a sharp transition.

##### Jams move backwards

Take the flow as a function of the local density, $q(\rho)$, and the conservation of cars in the continuum limit, a heuristic that the simulation then tests,

$$\partial_t \rho + \partial_x q(\rho) = 0.$$

A small perturbation of the density is carried at the speed $q'(\rho)$ (in sites per step). In free flow $q' = v_{\max} - p > 0$ and perturbations move with the cars. Above the maximum, $q'(\rho) < 0$: the perturbation moves *against* the cars. This is the start-stop wave: a car at the head of a jam leaves it, one at the back joins it, and the jam as a whole drifts backwards. How fast, is the subject of a section below. The space-time diagram shows it. At $\rho = 0.03$ the traffic is laminar. At $\rho = 0.10$, which is above the maximum, jams (orange: cars with $v = 0$) appear and travel upstream.

{% include figure.html src="/assets/img/2026-11-23-traffic/spacetime_p0.5_vmax5.png" caption="Space-time diagrams of a ring of L = 200 sites, time goes down. Left: density 0.03, free flow. Right: density 0.10, jams moving backwards. Orange: stopped cars." %}

> At $L = 100$ the same density $0.10$ is only 10 cars and shows no jam. This is the dependence on the size of the system, and it is the reason for $L = 200$ in the picture.

##### The case $v_{\max} = 1$

For $v_{\max} = 1$ every car has speed $1$ after rule 1, so the dynamics reduces to: a car moves if the site ahead is empty and it does not randomize. For a random sequential update every configuration is equally probable in the stationary state, the probability that the site ahead of a car is empty is $1 - \rho$ (up to terms of order $1/L$), and

$$q = (1 - p) \, \rho (1 - \rho),$$

as in [1]. The parallel update that I use gives a different function, found exactly by Schadschneider and Schreckenberg [3],

$$q = \frac{1}{2} \Big( 1 - \sqrt{1 - 4 (1 - p) \rho (1 - \rho)} \Big).$$

Two checks. For $p = 0$ it gives $\tfrac{1}{2}(1 - \lvert 1 - 2\rho \rvert) = \min(\rho, 1 - \rho)$, the bound above. And it is symmetric under $\rho \to 1 - \rho$, with maximum $(1 - \sqrt{p})/2$ at $\rho = 1/2$. 

##### The open road: a bottleneck

Now remove the periodic boundary. The paper [1] models a saturated two-lane road that feeds a one-lane road: when the leftmost site is empty, a car with $v = 0$ is put there, and the cars that reach the rightmost $v_{\max} + 1$ sites are deleted.

In a stationary state there are no sources or sinks of cars on the road. Hence the flow through every bond is the same, and equal to the rate of injection and to the rate of removal. The density is not a parameter any more: it comes out of $q = \rho \langle v \rangle$. I measure $q = 0.3032 \pm 0.0005$ and a bulk density $\rho = 0.0684$, to be compared with $0.304 \pm 0.001$ and $0.069 \pm 0.002$ in [1]. The ratio gives $\langle v \rangle = q/\rho = 4.43$, a little below the free-flow value $4.5$.

The point is a state that the *ring* also has. A ring at $\rho = 0.0680$ has $q = 0.3032 \pm 0.0001$ and at $\rho = 0.0685$ has $q = 0.3046 \pm 0.0001$, so the open road, at $\rho = 0.0684$ and $q = 0.3032$, lies on the fundamental diagram, on the free-flow side, below the maximum. This is the point the paper makes: the self-organized bottleneck state is *not* the state of maximum flow for $v_{\max} > 1$ [1]. For $v_{\max} = 1$ it is: the measured flow is $0.1463$ at $\rho = 0.498$, and the exact maximum is $(1 - \sqrt{0.5})/2 = 0.1464$ at $\rho = 1/2$.

The density profile shows that it is fixed at the entrance. Cars enter at rest, and a car at rest needs a few steps to leave site $0$, because of rule 3 with $p = 0.5$. A dense region forms there, and then dissolves into free flow:

| sites | 0-5 | 5-10 | 10-20 | 20-40 | 40-80 | 80 and beyond |
|---|---|---|---|---|---|---|
| density | 0.41 | 0.145 | 0.10 | 0.078 | 0.070 | 0.068 |

The mean speed in the first ten sites is $0.86$, against $4.4$ in the bulk. This looks like the outflow of a stationary jam: the cars leave a dense region at the rate at which they can restart. I have measured the profile, but I have not checked this identification directly. The speed of the jams, computed further below from this state, gives an indirect check: it works well where the jam is nearly packed, and not at small $p$.

The value depends on the parameters of the model, and not on the size of the system or on the seed: for $L$ between $200$ and $2000$ it stays in $0.301$-$0.3045$, and it is the same for different seeds. The two figures below show the dependence on $p$ and on $v_{\max}$.

{% include figure.html src="/assets/img/2026-11-23-traffic/bottleneck_vs_p_L1000_vmax5.png" caption="Open road, L = 1000, vmax = 5. Flow through the exit (orange) and density of the bulk (black) against p, one run of 100000 steps per point. The cross is the case of the paper." %}

The flow decreases with $p$, from $0.487$ at $p = 0.05$ to $0.304$ at $p = 0.5$ and $0.039$ at $p = 0.95$. The density of the bulk decreases too, as it must since $\rho = q / \langle v \rangle$. The last points, $p \geq 0.9$, are noisy in the density, and I would not read anything into them.

{% include figure.html src="/assets/img/2026-11-23-traffic/bottleneck_vs_vmax_L1000_p0.5.png" caption="Open road, L = 1000, p = 0.5. Flow through the exit (orange) and density of the bulk (black) against vmax." %}

The flow rises with $v_{\max}$ and saturates: $0.146, 0.248, 0.293$ for $v_{\max} = 1, 2, 3$, and between $0.303$ and $0.305$ for $4 \leq v_{\max} \leq 10$. This is consistent with the picture above: once the speed limit is high enough, the restart at the entrance is what limits the flow, and a higher limit only lowers the density of the free-flow region.

##### Two lanes, asymmetric lane changes

Rickert, Nagel, Schreckenberg and Latour [2] couple two rings, each one a single-lane model. The right lane is lane R, the left lane is lane L, and $N$ cars are distributed on the $2L$ sites. Each time step has two sub-steps:

- **a.** *Lane change.* Cars only move sideways, they do not advance. All the cars decide in parallel, on the configuration at the beginning of the step.
- **b.** *Single-lane update.* Rules 1 to 4, on each lane, with the configuration left by sub-step a.

A car at $x$ with speed $v$ changes lane if all of the following hold. The paper defines three lengths: how far to look ahead on one's own lane, ahead on the other lane, and back on the other lane. With $g$ the gap on the own lane, $g_o$ the number of empty sites ahead on the other lane and $g_b$ (b for back) the number of empty sites behind on the other lane, both counted from the site $x$ itself (so that they are $-1$ if the other lane is occupied at $x$):

- **T1.** $g < v + 1$. *Only for R to L.* This is the asymmetry: a car on the left lane does not need a reason, it always tries to return to the right lane.
- **T2.** $g_o > v + 1$: enough room ahead on the other lane.
- **T3.** $g_b > v_{\max}$: enough room behind on the other lane.
- **T4.** a random number is below $p_{\mathrm{change}}$. The paper uses $p_{\mathrm{change}} = 1$.

Let me check what these conditions guarantee.

- **No collisions in the lane change.** Two cars of different lanes at the same site $x$ see $g_o = g_b = -1$, which is never larger than $v + 1$ or $v_{\max}$, so neither of them changes. Two cars of the same lane cannot move into the same site because they are at different sites. So after sub-step a no two cars share a cell, and sub-step b then works as for the single lane.
- **The car that changes is not slowed down.** After the change its gap ahead is $g_o \geq v + 2$, larger than $v$, so rule 1 and rule 2 do not stop it from accelerating.
- **The car behind it is not forced to brake.** The new gap behind is $g_b \geq v_{\max} + 1$. The follower has speed at most $v_{\max}$, so rule 2, $v \leftarrow \min(v, g)$, does not change its speed: the newcomer does not disturb the lane it enters. This is the role of T3, the look-back. The paper shows that setting it to zero destroys the laminar flow and lowers the maximum of the flow [2].

The density and the flows are defined per lane. With $N_R, N_L$ the number of cars on each lane and $v_i$ their speeds,

$$\begin{aligned}
\rho &= \frac{N}{2L} = \frac{\rho_R + \rho_L}{2}, \\
j_R &= \frac{1}{L} \sum_{i \in R} v_i, \qquad j_L = \frac{1}{L} \sum_{i \in L} v_i, \\
\bar{q} &= \frac{j_R + j_L}{2} = \frac{1}{2L} \sum_{i} v_i,
\end{aligned}$$

where $\rho_R = N_R/L$ and $\rho_L = N_L/L$ are the densities of the lanes. The bound above holds in the same way, $\bar{q} \leq \min(v_{\max}\rho, 1 - \rho)$, since after the lane change every car still has $v \leq g$ on its lane.

{% include figure.html src="/assets/img/2026-11-23-traffic/fundamental_diagram_2lanes_L1000_p0.5_vmax5.png" caption="Two lanes, asymmetric, L = 1000 sites per lane, p = 0.5, vmax = 5, p_change = 1. Flow per lane against the density per lane: right lane (black), left lane (hollow), and the two together (orange)." %}

At low density the left lane is almost empty. A car moves to the left only if there is another car within $v + 1 \leq 6$ sites ahead of it. For Poisson-distributed cars the probability of that is about $6 \rho_R$, and the car returns after a short time, so $\rho_L = O(\rho^2)$ as $\rho \to 0$: this mean-field argument is the one of the paper [2], and it explains why the left-lane flow starts well below the one of the right lane. Above $\rho \approx 0.4$ the two lanes carry about the same flow.

{% include figure.html src="/assets/img/2026-11-23-traffic/comparison_1lane_2lanes_L2000_p0.5_vmax5.png" caption="Zoom near the maximum, L = 2000 sites per lane, 50000 steps per point. Single lane (black) and two lanes, asymmetric, per lane (orange)." %}

Near the maximum the two lanes carry more *per lane* than a single one: about $0.345$ at $\rho = 0.08$ against about $0.325$ at $\rho = 0.075$. The paper states the same (point (i) of Sect. 4.2: the maximum flow of two lanes is higher than twice the one of the single lane) [2]. The lane changes remove some of the disturbances of the single lane, and the effect is beneficial although they are themselves disturbances.

{% include figure.html src="/assets/img/2026-11-23-traffic/lane_changes_L1000_p0.5_vmax5.png" caption="Lane changes per site and time step (left) and per car and time step (right), against the density per lane." %}

The number of lane changes per site has its maximum at $\rho \approx 0.25$, much above the density of maximum flow, as in the paper. Per car it is largest at low density, around $\rho \approx 0.04$, below the density of maximum flow, again as in the paper.

{% include figure.html src="/assets/img/2026-11-23-traffic/spacetime_2lanes_p0.5_vmax5.png" caption="Space-time diagram of the two lanes at density 0.09 per lane, a window of 400 sites and 300 steps from a ring of 2000 sites per lane. Orange: stopped cars." %}

At $\rho = 0.09$, slightly above the maximum, jams appear in both lanes and travel backwards, as in the single lane, and the two lanes are strongly correlated through the lane changes.

The paper itself points out that most of the lane changes in this model are an artefact of the rules, a *tailgating dance*: a follower at speed $5$ sees a car $5$ sites ahead, moves to the left, and next step, when the gap has grown to $6$, comes back [2].

##### How fast does a jam move?

The jam in the space-time diagrams is a travelling structure: a packed region that moves upstream at a fixed speed while the cars pass through it. Here the speed can be computed from the fundamental diagram and measured in the simulation, and the two can be compared.

**The formula.** Let an edge between two uniform states move at velocity $v_J$. On its left the density and the flow are $(\rho_1, q_1)$, on its right $(\rho_2, q_2)$. Take two fixed points $a < X(t) < b$, with $a$ on the left and $b$ on the right, and count the cars between them,

$$N(t) = \rho_1 \, (X - a) + \rho_2 \, (b - X), \qquad \frac{dN}{dt} = v_J \, (\rho_1 - \rho_2).$$

The cars only enter at $a$ and leave at $b$, so $dN/dt = q_1 - q_2$, and

$$v_J = \frac{q_2 - q_1}{\rho_2 - \rho_1}.$$

For a small jump this is $q'(\rho)$, the speed of the previous section. For a large one it is the slope of the chord between the two states on the fundamental diagram. Now take the *front* of the jam, where the cars leave. On the left (upstream) is the jam itself, on the right the cars that have left it, in a state $(\rho_{\text{out}}, q_{\text{out}})$. If the jam is packed, with $q_j \approx 0$ and $\rho_j \approx 1$,

$$v_J = \frac{q_{\text{out}} - q_j}{\rho_{\text{out}} - \rho_j} \approx -\frac{q_{\text{out}}}{1 - \rho_{\text{out}}}.$$

The cars leave the jam at the flow $q_{\text{out}}$, each of them frees about one site, and the front retreats by that many sites per step. Nothing in the formula depends on the overall density of the road. What is left is the outflow state. The natural candidate is the bottleneck of the open road, which is also a state of cars that start from rest: for $p = 0.5$, $v_{\max} = 5$ it gives $q_{\text{out}} = 0.304$, $\rho_{\text{out}} = 0.068$, and

$$v_J \approx -\frac{0.304}{1 - 0.068} = -0.326 \ \text{sites per step.}$$

**The measurement.** I take the pattern of the stopped cars ($v = 0$) in space and time. If the pattern at time $t + \tau$ is the one at time $t$ shifted by $k$ sites, the jams move at $k / \tau$. I find $k$ with the circular cross-correlation of the two patterns, averaged over time, for four lags $\tau$ between $50$ and $200$ steps, and I fit $k = v_J \tau$ through the origin. All the jams of the ring move together, so they all contribute. For two lanes the pattern is the sum of the stopped cars of the two lanes. The roads have $L = 2000$ sites, and each point is the average of three runs.

{% include figure.html src="/assets/img/2026-11-23-traffic/jam_speed_vs_density_p0.5_vmax5.png" caption="Speed of the jams against the density per lane, p = 0.5, vmax = 5, L = 2000. Single lane (black), two lanes (orange), and the prediction from the bottleneck (dashed). Negative: backwards. Below density 0.07 there are no jams." %}

Against the density the speed is almost constant: from $-0.340$ at $\rho = 0.10$ to $-0.369$ at $\rho = 0.58$, about $10\%$ over a sixfold change of density. The prediction, $-0.326$, is close but not on the points. Jams do not exist below $\rho \approx 0.07$, where there are no stopped cars at all.

{% include figure.html src="/assets/img/2026-11-23-traffic/jam_speed_vs_p_rho0.4_vmax5.png" caption="Speed of the jams against the slowdown probability p, density 0.4 per lane, vmax = 5. The dashed line is the prediction from the bottleneck of the open road for each p." %}

The dependence on $p$ is strong: $-0.80$ at $p = 0.1$, $-0.354$ at $p = 0.5$, $-0.081$ at $p = 0.9$. A car leaves the jam with probability $1 - p$ at each attempt, so the jam retreats faster when drivers hesitate less. The prediction follows the points for $p \geq 0.6$ (for example $-0.221$ measured and $-0.204$ predicted at $p = 0.7$) and underestimates the speed at small $p$: at $p = 0.1$ it gives $-0.52$ against the measured $-0.80$.

{% include figure.html src="/assets/img/2026-11-23-traffic/jam_speed_vs_vmax_rho0.4_p0.5.png" caption="Speed of the jams against the maximum speed vmax, density 0.4 per lane, p = 0.5." %}

The dependence on $v_{\max}$ is weak. For the single lane it is $-0.322$ at $v_{\max} = 2$ and between $-0.350$ and $-0.354$ from $v_{\max} = 3$ to $10$. Once the cars in the jam are stopped, a higher speed limit has nothing to do with how they restart. The prediction has the opposite small trend, from $-0.339$ to $-0.315$, and differs from the points by up to $0.04$.

The single lane and the two lanes are close in every figure, within $0.01$ to $0.02$ (the two lanes are a little faster at moderate density and at $v_{\max} = 3$ to $5$). I did not test why. My guess is that at these densities both lanes are jammed, so that changing lane does not help a stopped car.

> About the uncertainty. The error bars in the figures are only the scatter between the three runs, about $0.002$. They do not include the effect of the method: the speed obtained from different lags differs by about $0.015$, for example $-0.331$, $-0.312$ and $-0.313$ for $\tau = 100, 200, 300$ at $\rho = 0.10$, because the jams change shape while they move. Differences between the curves below about $0.02$ should therefore not be read as real.

##### The implementation

The code is in two files, `model.py` for the single lane and `model_multiple.py` for the two lanes, each with a test file that compares the vectorized rules with an independent implementation written car by car.

| Rule | Code |
|---|---|
| gap $g_i$ | `SingleLane.gaps` (with $d = g + 1$) |
| rules 1-4 | `SingleLane.update` |
| flow $q = \rho \langle v \rangle$, errors | `SingleLane.measure` (relaxation of $10L$ steps, errors from block averages) |
| open boundary | `SingleLane(bc='open')`, `SingleLane.update` |
| $g_o$, $g_b$ | `TwoLane.gaps_other` |
| T1 to T4 | `TwoLane.lane_change` |
| sub-step b | `TwoLane._advance` |
| $j_R$, $j_L$, $\bar{q}$, lane changes | `TwoLane.measure` |
| speed of the jams | `jam_speed.pattern_velocity` (cross-correlation of the stopped cars), `jam_speed.predicted` |

All the figures are made by `plots.py`, `plots_multiple.py` and `jam_speed.py`.

Some remarks:
>The errors are not $\sigma/\sqrt{T}$. Successive time steps are strongly correlated, because a jam lives for many steps, so I take the scatter of twenty block averages instead.

>The relaxation matters. Starting from all cars at rest, a measurement without it mixes the transient with the stationary state; the paper discards $10L$ steps for the single lane, and so do I.

Cheers.

> References
>1. K. Nagel, M. Schreckenberg. *A cellular automaton model for freeway traffic.* Journal de Physique I 2(12):2221-2229, 1992. <https://hal.science/jpa-00246697>
>2. M. Rickert, K. Nagel, M. Schreckenberg, A. Latour. *Two lane traffic simulations using cellular automata.* Physica A 231:534-550, 1996. <https://arxiv.org/abs/cond-mat/9512119>
>3. A. Schadschneider, M. Schreckenberg. *Cellular automaton models and traffic flow.* Journal of Physics A 26:L679-L683, 1993.
