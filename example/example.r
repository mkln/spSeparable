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


# separable data generation (matrix normal)
Sigma <- solve(rWishart(1, q+1, diag(q))[,,1])
Omega <- cov2cor(Sigma)
A <- t(chol(Sigma))
custom_dag <- spiox::dag_vecchia(coords, 15, TRUE)

phi <- 30

# intercept + 2 covariates only
p <- 3 
X <- matrix(1, ncol=1, nrow = nr) %>%
  cbind(matrix(rnorm(nr*(p-1)), ncol=p-1))

Beta <- matrix(rnorm(q * p), ncol=q)

# sample separable (matrix normal) data
U <- rnorm(nr * q) %>% matrix(ncol=q)
C <- exp(- phi * as.matrix(dist(coords)))
Ci <- solve(C)
H <- chol(Ci)
L <- t(chol( C ))
Y <- X %*% Beta + L %*% U %*% t(A)


m_nn <- 20
mcmc <- 2000

# import this function from spiox package
if (!requireNamespace("spiox", quietly = TRUE)) remotes::install_github("mkln/spiox")
custom_dag <- spiox::dag_vecchia(coords, m_nn, TRUE)

# this uses the same interface as spiox
# fix everything but the first row here to do so
theta <- c(50, 1, 0.5, 0)
sample_theta <- c(1, 0, 0, 0)

system.time({
  spsep_out <- spseparable_response(Y, coords, custom_dag, 
                                    theta_start = theta,
                                    sampling = sample_theta,
                                    Sigma_start = Sigma, 
                                    X = X, # X = NULL vec(Y) ~ N(0, R %x% Sigma)
                                    mcmc,
                                    print_every=100,
                                    dag_opts=-1,
                                    upd_Sigma = T,
                                    upd_theta = TRUE,
                                    num_threads = n_threads) })

# plot theta and diag(Sigma)
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
Omega_sep <- spsep_out$Sigma %>% apply(3, \(s) cov2cor(s)) %>% array(dim=c(q,q,mcmc))

Omega_sep[2,3,] %>% plot(type='l')

Omega_sep %>% apply(1:2, mean)

# Beta trace plot
plot_spsep_beta <- function(spsep_out, perc_show=0.75){
  idx   <- (floor((1-perc_show) * mcmc) + 1):mcmc # last 50% for inference
  
  # diagonals across iterations
  beta_mat <- spsep_out$Beta[, , idx]
  
  df_beta <- as.data.frame.table(beta_mat, responseName = "value")
  # Rename dims and coerce to ints
  df_beta <- df_beta %>%
    rename(beta = Var1, outcome = Var2, iter = Var3) %>%
    mutate(beta = factor(beta, labels = paste0("beta", 1:nlevels(beta))),
           outcome = factor(outcome, labels = paste0("y", 1:nlevels(outcome))),
           iter = as.integer(iter))
  
  truth_beta <- as.data.frame.table(Beta, responseName = "true") %>%
    rename(beta = Var1, outcome = Var2) %>%
    mutate(beta    = factor(beta, labels = paste0("beta", 1:nlevels(beta))),
           outcome = factor(outcome, labels = paste0("y", 1:nlevels(outcome))))
  
  iter_max <- max(df_beta$iter)
  truth_beta <- mutate(truth_beta, iter_max = iter_max)
  
  # Traceplots: rows = beta, cols = outcome
  ggplot(df_beta, aes(x = iter, y = value)) +
    geom_line(linewidth = 0.25) +
    facet_grid(beta ~ outcome, scales = "free_y") +
    # add the true value line per facet
    geom_hline(data = truth_beta,
               aes(yintercept = true),
               linetype = "dashed", color = "red") +
    # optional: label the true value at the right edge
    geom_text(data = truth_beta,
              aes(x = iter_max, y = true, label = sprintf("%.2f", true)),
              inherit.aes = FALSE, hjust = 1.05, vjust = -0.4,
              size = 3, color = "red") +
    coord_cartesian(xlim = c(min(df_beta$iter), iter_max * 1.02)) +
    labs(x = "Iteration", y = "Value",
         title = "Traceplots for Beta by outcome",
         subtitle = "Reference line and value = true Beta") +
    theme_bw() 
}

spsep_out %>% plot_spsep_beta(0.5)

# Beta posterior mean
spsep_out$Beta %>% apply(c(1:2), mean) %>% round(2)
