---
layout: post
title: "And on the Seventh Sample, We Divided by n−1"
category: stat
---
> You are told to estimate the variance by dividing by $n-1$ instead of $n$. The usual justification is a hand-wave about "losing a degree of freedom". Let me try to formalize it.

Let $X_1, \dots, X_n$ be iid random variables with unknown mean $\mu$ and variance $\sigma^2$. Define the sample mean and the two candidate estimators of the variance
$$
\begin{aligned}
\bar X &= \frac{1}{n} \sum_{i=1}^{n} X_i, \\
S_n^2 &= \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar X)^2, \\
S_{n-1}^2 &= \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar X)^2.
\end{aligned}
$$

**Assumptions.** Only $\mathbb{E}[X_i] = \mu$, $\mathrm{Var}(X_i) = \sigma^2 < \infty$ and independence (uncorrelated is enough). No Gaussianity is needed. In what follows, $\mathbb{E}$ denotes the expectation over the random sample.

> Let's prove that $\mathbb{E}[S_n^2] \neq \sigma^2$, and that $\mathbb{E}[S_{n-1}^2] = \sigma^2$.

##### Step 1: recentre on the true mean

The deviations from $\bar X$ are awkward, because $\bar X$ itself depends on the sample. Write $X_i - \bar X = (X_i - \mu) - (\bar X - \mu)$ and expand the square:

$$\sum_{i=1}^{n} (X_i - \bar X)^2 = \sum_{i=1}^{n} (X_i - \mu)^2 - 2(\bar X - \mu) \sum_{i=1}^{n} (X_i - \mu) + n (\bar X - \mu)^2.$$

Since $\sum_i (X_i - \mu) = n(\bar X - \mu)$, the last two terms combine and we get the identity
$$\sum_{i=1}^{n} (X_i - \bar X)^2 = \sum_{i=1}^{n} (X_i - \mu)^2 - n (\bar X - \mu)^2.$$

This is the key point. The sum of squares around the *sample* mean is always smaller than the sum around the *true* mean, by exactly $n(\bar X - \mu)^2 \geq 0$. The sample mean is the point that minimizes the sum of squares, and it is tuned to this very sample.

##### Step 2: take the expectation

By definition of the variance, $\mathbb{E}[(X_i - \mu)^2] = \sigma^2$, so the first term gives $n\sigma^2$. For the second term, $\bar X$ is an average of $n$ independent variables with mean $\mu$, hence
$$\mathbb{E}[(\bar X - \mu)^2] = \mathrm{Var}(\bar X) = \frac{1}{n^2} \sum_{i=1}^{n} \mathrm{Var}(X_i) = \frac{\sigma^2}{n}.$$

Putting the pieces together,
$$\mathbb{E}\left[ \sum_{i=1}^{n} (X_i - \bar X)^2 \right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\,\sigma^2.$$

##### Result

Dividing by $n$ we find
$$\mathbb{E}[S_n^2] = \frac{n-1}{n}\,\sigma^2 = \sigma^2 - \frac{\sigma^2}{n},$$

so $S_n^2$ is biased: it underestimates the variance by a relative amount $1/n$. Dividing by $n-1$ instead removes the bias exactly,
$$\mathbb{E}[S_{n-1}^2] = \frac{n}{n-1}\,\mathbb{E}[S_n^2] = \sigma^2.$$

>The bias is $-\sigma^2/n$, which is the variance of the sample mean. In other words, the missing piece is exactly the error we make by using $\bar X$ in place of $\mu$.

{% include figure.html src="/assets/img/2026-11-18-bessel-variance/bias.png" caption="Mean of the two estimators against the sample size n, for a true variance σ² = 4. The 1/n estimator sits at (n−1)/n · σ², i.e. at 2 for n = 2, and approaches σ² only slowly." %}

The plot comes from a simple Monte Carlo experiment: for each $n \in \{2, 3, 5, 10, 30, 100, 1000\}$ I draw $200000$ independent samples of size $n$ from $\mathcal{N}(0, \sigma^2)$ with $\sigma = 2$, compute both estimators on each sample, and average over the samples. The plot agrees with the formula. At $n = 2$ the biased estimator averages $\frac{1}{2}\sigma^2$, which is half of the truth, while $S_{n-1}^2$ is on the line for every $n$. Both estimators agree for large $n$, since the factor $\frac{n}{n-1} \to 1$.

Some remarks:
>Where does the $n-1$ come from? The $n$ residuals $X_i - \bar X$ are not independent: they satisfy the linear constraint $\sum_i (X_i - \bar X) = 0$. Only $n-1$ of them are free, and this is the "lost degree of freedom". For Gaussian data this is made precise by $(n-1) S_{n-1}^2 / \sigma^2 \sim \chi^2_{n-1}$.

>If the mean $\mu$ is known, there is nothing to correct. By Step 1, the estimator $\frac{1}{n}\sum_i (X_i - \mu)^2$ has expectation exactly $\sigma^2$, and you should divide by $n$. The correction is the price of estimating $\mu$ from the same data.

>There is a way to see the result without $\bar X$ at all. The identity $\sum_{i<j} (X_i - X_j)^2 = n \sum_i (X_i - \bar X)^2$ gives $S_{n-1}^2 = \frac{1}{n(n-1)} \sum_{i<j} (X_i - X_j)^2$. Each pair has $\mathbb{E}[(X_i - X_j)^2] = 2\sigma^2$, and there are $n(n-1)/2$ pairs, so the average works out to $\sigma^2$ exactly. The $n$ "diagonal" pairs $i = j$ are always zero and must not be counted.

>Unbiased does not mean best. $S_{n-1}^2$ is unbiased for $\sigma^2$, but its square root $S_{n-1}$ is still a biased estimator of $\sigma$: since $\sqrt{\cdot}$ is concave, Jensen's inequality gives $\mathbb{E}[S_{n-1}] \leq \sigma$. And for Gaussian data the estimator that minimizes the mean squared error divides by $n+1$, trading a little bias for a smaller variance.

Cheers.
