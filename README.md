# GP-LMC (separable or intrinstic specification) Response Models

This package implements Gaussian Process response models for multivariate spatial data with efficient computation based on Vecchia and DAG-based GP approximations. The cross-covariance specification is a separable or intrinsic specification, which is too simplistic in many research applications but remains useful for comparisons.

The main entry point is the function `spseparable_response()`, which runs an adaptive Metropolis MCMC sampler for the latent GP hyperparameters and a conditionally conjugate Inverse Wishart update for the coregionalization matrix.

Find an example in [`example.r`](example/example.r)

## Math details

We observe a data matrix ![Y](https://latex.codecogs.com/svg.latex?Y\in\mathbb{R}^{n\times%20q}), where ![n](https://latex.codecogs.com/svg.latex?n) is the number of spatial locations and ![q](https://latex.codecogs.com/svg.latex?q) is the number of outcomes.  
Let ![vec(Y)](https://latex.codecogs.com/svg.latex?\mathrm{vec}(Y)) denote the column-stacked vectorization of ![Y](https://latex.codecogs.com/svg.latex?Y).

We assume

![likelihood](https://latex.codecogs.com/svg.latex?\mathrm{vec}(Y)\sim%20N\left(0,R\otimes\Sigma\right))

which is the same as a matrix Normal distribution on ![Y](https://latex.codecogs.com/svg.latex?Y) with row covariance ![R](https://latex.codecogs.com/svg.latex?R) and column covariance ![Sigma](https://latex.codecogs.com/svg.latex?\Sigma), where  

- ![R](https://latex.codecogs.com/svg.latex?R\in\mathbb{R}^{n\times%20n}) is the spatial correlation matrix (e.g., Matérn),  
- ![Sigma](https://latex.codecogs.com/svg.latex?\Sigma\in\mathbb{R}^{q\times%20q}) is the cross-outcome covariance matrix.

---

This package fits models with measurement error within the spatial correlation structure, **does not** allow setting the number of factors to less than the number of outcomes, and **does not** estimate linear covariate effects.  

For these additional features, consider the R packages [`meshed`](https://github.com/mkln/meshed) and [`spBayes`](https://cran.r-project.org/package=spBayes). For the non-separable LMC, use [`spLMC`](https://github.com/mkln/spLMC). For IOX, use [`spiox`](https://github.com/mkln/spiox).

