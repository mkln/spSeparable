# GP-LMC (separable or intrinstic specification) Response Models

This package implements Gaussian Process Linear Model of Coregionalization (GP-LMC) response models (separable specification) with efficient computation based on Vecchia and DAG-based GP approximations.

The main entry point is the function `sepmv_response()`, which runs an adaptive Metropolis MCMC sampler for the latent GP hyperparameters and the factor loadings matrix.

Find an example in [`example.r`](example/example.r)

## Math details

We observe a data matrix ![Y](https://latex.codecogs.com/svg.latex?Y\in\mathbb{R}^{n\times%20q}), where ![n](https://latex.codecogs.com/svg.latex?n) is the number of spatial locations and ![q](https://latex.codecogs.com/svg.latex?q) is the number of outcomes.  
Let ![vec(Y)](https://latex.codecogs.com/svg.latex?\mathrm{vec}(Y)) denote the column-stacked vectorization of ![Y](https://latex.codecogs.com/svg.latex?Y).

We assume

![likelihood](https://latex.codecogs.com/svg.latex?\mathrm{vec}(Y)\sim%20N\left(0,R\otimes K\right))

where  

- ![R](https://latex.codecogs.com/svg.latex?R\in\mathbb{R}^{n\times%20n}) is the spatial correlation matrix (e.g., Matérn),  
- ![D](https://latex.codecogs.com/svg.latex?D=\tau^2I_n) is the nugget (independent noise) term,
- ![K](https://latex.codecogs.com/svg.latex?K\in\mathbb{R}^{q\times%20q}) is the cross-outcome covariance matrix.

---

This package fits models with measurement error within the spatial correlation structure, **does not** allow setting the number of factors to less than the number of outcomes, and **does not** estimate linear covariate effects.  

For these additional features, consider the R packages [`meshed`](https://github.com/mkln/meshed) and [`spBayes`](https://cran.r-project.org/package=spBayes).