---
layout: post
title: "And there was evening, and there was 12, the first day"
category: nn
---
> If you've trained a language model with a $\approx 150000$-token vocabulary, you've seen the first step's cross-entropy loss land around $12$ (roughly $11.9$–$12.1$). That seems puzzling at first, but it actually makes sense, and it is a useful sanity check. Let me try to formalize it.

Let $V = 150000$ be the vocabulary size. For a context with final hidden state $h \in \mathbb{R}^d$ and unembedding matrix $W \in \mathbb{R}^{V \times d}$, the logits are $z = Wh$. The model's distribution over the vocabulary is $p = \operatorname{softmax}(z)$, i.e. $p_i = e^{z_i} / \sum_j e^{z_j}$, and the per-token cross-entropy loss for target $y$ is
$$L_{xent} = -\log(p_y) = \operatorname{LSE}(z) - z_y, \quad \text{where} \quad \operatorname{LSE}(z) = \log \left( \sum_j e^{z_j} \right).$$

**Assumptions.** Assume $W_{ij} \sim \text{iid} \, \mathcal{N}(0, \sigma^2)$, independent of both $h$ and the target $y$. (This holds for an untied unembedding matrix; with tied embeddings, $W$ also produces $h$ and the argument below does not apply as is.) Conditioning on $h$, the logits are then $\text{iid} \, \mathcal{N}(0, s^2)$ with $s^2 = \sigma^2 \lVert h \rVert^2$. If the last layer is a LayerNorm/RMSNorm with unit gain, then $\lVert h \rVert^2 \approx d$. For example, with $\sigma = 0.02$ and $d = 768$ we get $s^2 \approx 0.31$.

In what follows, $\mathbb{E}$ denotes the expectation over the random initialization $W$, for fixed $h$ and $y$.

> Let's prove that we can extract bounds on the expectation value $\mathbb{E}[L_{xent}]$.

Since the target $y$ is independent of $W$, $z_y \sim \mathcal{N}(0, s^2)$ and $\mathbb{E}[z_y] = 0$. Hence
$$\mathbb{E}[L_{xent}] = \mathbb{E}[\operatorname{LSE}(z)] - \mathbb{E}[z_y] = \mathbb{E}[\operatorname{LSE}(z)].$$

##### Lower bound

Write $\operatorname{LSE}(z) = \log\left( V \cdot \frac{1}{V} \sum_j e^{z_j} \right) = \log V + \log\left( \frac{1}{V} \sum_j e^{z_j} \right)$.

Since $\log$ is concave and the weights $1/V$ are nonnegative and sum to $1$, the log of an average is at least the average of the logs. For every fixed $z$ we have

$$\log\left( \frac{1}{V} \sum_j e^{z_j} \right) \geq \frac{1}{V} \sum_j \log(e^{z_j}) = \frac{1}{V} \sum_j z_j.$$

Now take the expectation over the random initialization. The right-hand side has mean $\frac{1}{V} \sum_j \mathbb{E}[z_j] = 0$, since each $z_j \sim \mathcal{N}(0, s^2)$. Taking expectations of both sides therefore gives $\mathbb{E}\left[\log\left( \frac{1}{V} \sum_j e^{z_j} \right)\right] \geq 0$, and so
$$\mathbb{E}[L_{xent}] = \mathbb{E}[\operatorname{LSE}(z)] \geq \log V.$$

Note that this bound only uses $\mathbb{E}[z_j] = 0$, not Gaussianity.

##### Upper bound

Let $X = \frac{1}{V} \sum_j e^{z_j}$. Using again that $\log$ is concave, Jensen's inequality over the random initialization gives $\mathbb{E}[\log X] \leq \log(\mathbb{E}[X])$. By linearity and the log-normal mean $\mathbb{E}[e^{z_j}] = e^{s^2/2}$ (the Gaussian moment generating function at $1$), we have $\mathbb{E}[X] = e^{s^2/2}$, so
$$\mathbb{E}[\log X] \leq \log\left(e^{s^2/2}\right) = \frac{s^2}{2}.$$

Hence, using $\mathbb{E}[L_{xent}] = \mathbb{E}[\operatorname{LSE}(z)] = \log V + \mathbb{E}[\log X]$,
$$\log V \leq \mathbb{E}[L_{xent}] \leq \log V + \frac{s^2}{2}.$$

>$\log(150000) \approx 11.92$. With $s^2 \approx 0.31$, the correction is at most $s^2/2 \approx 0.15$, so $\mathbb{E}[L_{xent}] \in [11.92,\, 12.07]$. That is the "$\approx 12$" observed in practice.

The lower bound ignores the variance of the logits, so it is loose by about $s^2/2$. The upper bound is nearly exact. Since the $z_j$ are iid given $h$, we have $\mathrm{Var}(e^{z_j}) = e^{s^2}(e^{s^2}-1)$ and $\mathbb{E}[e^{z_j}]^2 = e^{s^2}$. Averaging $V$ iid terms divides the variance by $V$, so
$$\frac{\mathrm{Var}(X)}{\mathbb{E}[X]^2} = \frac{e^{s^2}-1}{V}.$$
This is small when $e^{s^2} \ll V$, i.e. when $s^2 \ll \log V$. In that regime $X$ concentrates around its mean, and a second-order Taylor expansion of $\log$ around $\mathbb{E}[X]$ gives
$$\mathbb{E}[\log X] \approx \frac{s^2}{2} - \frac{e^{s^2}-1}{2V}.$$
Here the correction is of order $10^{-6}$, so $\mathbb{E}[L_{xent}] \approx \log V + s^2/2 \approx 12.07$ is essentially exact, not just an upper bound. 

Some remarks:
>If $z = 0$, then $p$ is exactly uniform and $L_{xent} = \log V$ for every target. More generally, the cross-entropy of any target distribution against a uniform model equals $\log V$, which is the entropy of the uniform distribution. The loss itself is not capped at $\log V$: a confidently wrong model can exceed it. Small random logits perturb the uniform value by a term of order $s^2$.

>The result is for the expected loss of a single token. A single token's loss fluctuates with standard deviation about $s \approx 0.55$, and the loss you see at step 0 is an average over many tokens.

>If the initial loss is much larger than $\log V + s^2/2$, the effective logit scale is larger than intended. Inverting the formula, $s^2_{\text{eff}} \approx 2(L_0 - \log V)$, which you can compare with $\sigma^2 d$ (for example, a missing final norm or outlier hidden dimensions). Once $s^2$ approaches $2\log V$, the loss grows roughly linearly in $s$ rather than quadratically.

Cheers.

{% include figure.html src="/assets/img/2026-11-18-random-loss/face.jpg"%}