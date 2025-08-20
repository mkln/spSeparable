library(tidyverse)
library(spSeparable)

# spiox package required to run dag_vecchia() 
# remotes::install_github("mkln/spiox")

set.seed(25) 
n_threads <- 16

q <- 6
# spatial coordinates
coords <- expand.grid(x <- seq(0, 1, length.out=20), x) %>%
  as.matrix()
colnames(coords) <- c("Var1","Var2")
nr <- nrow(coords)


# lmc data
Sigma <- solve(rWishart(1, q+1, diag(q))[,,1])
A <- t(chol(Sigma))
custom_dag <- spiox::dag_vecchia(coords, 15, TRUE)

phi <- 30

# sample separable (matrix normal) data
U <- rnorm(nr * q) %>% matrix(ncol=q)
C <- exp(- phi * as.matrix(dist(coords)))
Ci <- solve(C)
H <- chol(Ci)
L <- t(chol( C ))
Y <- L %*% U %*% t(A)


m_nn <- 40
mcmc <- 2000

# import this function from spiox package
if (!requireNamespace("spiox", quietly = TRUE)) remotes::install_github("mkln/spiox")
custom_dag <- spiox::dag_vecchia(coords, m_nn, TRUE)

# this uses the same interface as spiox
# fix everything but the first row here to do so
theta <- c(50, 1, 0.5, 0)
sample_theta <- c(1, 0, 0, 0)

spseparable_logdens(Y, coords, custom_dag, theta, Sigma)

MN::dmn(Y, matrix(0, nr, q), U=exp(- 50 * as.matrix(dist(coords))), V=Sigma, logged = T)

# Matrix-normal log-density using row whitener H (t(H) %*% H = R^{-1})
# and column whitener AinvT (solve(chol(Sigma))) with known log|R^{-1}|.
logdmatrixnormal <- function(X, M, H, AinvT, logdetRinv) {
  n <- nrow(X); q <- ncol(X)
  if (!all(dim(M) == c(n, q))) stop("M must be n x q")
  if (!all(dim(H) == c(n, n))) stop("H must be n x n")
  if (!all(dim(AinvT) == c(q, q))) stop("AinvT must be q x q")
  Z <- H %*% (X - M) %*% AinvT                     # = (I⊗H) vec(X-M) then column whiten
  quad <- sum(Z * Z)                               # Frobenius norm squared
  logdetSigma <- -2 * sum(log(diag(AinvT)))        # since AinvT = chol(Sigma)^{-1}

  colc <- -n/2 * logdetSigma
  rowc <- 0.5 * q * logdetRinv
  cc <- -n * q * log(2 * pi)/2
  print(c(cc, rowc, colc, -0.5*quad))
  #return(-0.5 * (n * q * log(2 * pi) + n * logdetSigma) + 0.5 * q * logdetRinv - 0.5 * quad)
  return(cc + rowc + colc - 0.5 * quad)
}

# Zero-mean convenience
logdmatrixnormal0 <- function(X, H, AinvT, logdetRinv) {
  logdmatrixnormal(X, M = matrix(0, nrow(X), ncol(X)), H = H, AinvT = AinvT, logdetRinv = logdetRinv)
}
logdmatrixnormal0(Y, H, solve(chol(Sigma)), 2*sum(log(diag(H))))








system.time({
  spsep_out <- spseparable_response(Y, coords, custom_dag, theta, sample_theta,
                                    Sigma, 
                                    mcmc,
                                    print_every=100,
                                    dag_opts=-1,
                                    upd_Sigma = T,
                                    upd_theta = TRUE,
                                    num_threads = n_threads) })

plot_spsep <- function(spsep_out, perc_show=0.75){
  q     <- dim(spsep_out$Sigma)[1]
  mcmc  <- dim(spsep_out$Sigma)[3]
  idx   <- (floor((1-perc_show) * mcmc) + 1):mcmc  # last 3/4
  
  # diagonals across iterations
  sigma_mat <- sapply(1:q, function(j) spsep_out$Sigma[j, j, idx])
  phi_mat <- spsep_out$theta[1, idx]
  
  df_sigma <- as.data.frame(sigma_mat) |>
    mutate(iter = idx) |>
    pivot_longer(-iter, names_to = "j", values_to = "value") |>
    mutate(param = "sigma2", j = as.integer(gsub("\\D", "", j)))
  
  df_phi <- as.data.frame(phi_mat) |>
    mutate(iter = idx) |>
    pivot_longer(-iter, names_to = "j", values_to = "value") |>
    mutate(param = "phi", j = as.integer(gsub("\\D", "", j)))
  
  plotted0 <- ggplot(df_phi, aes(iter, value)) +
    geom_line(linewidth = 0.3) +
    facet_wrap(param ~ j, scales = "free_y", ncol=q) +
    labs(x = "Iteration", y = "Value") +
    theme_minimal()
  
  plotted1 <- ggplot(df_sigma, aes(iter, value)) +
    geom_line(linewidth = 0.3) +
    facet_wrap(param ~ j, scales = "free_y", ncol=q) +
    labs(x = "Iteration", y = "Value") +
    theme_minimal()
  
  return(gridExtra::grid.arrange(plotted0, plotted1, nrow=2))
}

spsep_out %>% plot_spsep(0.5)

# zero distance correlation
Omega_sep <- spspsep_out$Sigma %>% apply(3, \(s) cov2cor(s)) %>% array(dim=c(q,q,mcmc))

Omega_sep[2,3,] %>% plot(type='l')
