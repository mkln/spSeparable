#include "spseparable.h"
#include "interrupt.h"

using namespace std;

//' Fit a Separable Gaussian Process Response Model
//'
//' Implements an adaptive Metropolis-within-Gibbs sampler for the separable
//' multivariate Gaussian process response model.
//' The model assumes
//' \deqn{ \mathrm{vec}(Y) \sim N\left(X \beta, R \otimes \Sigma \right) }
//' where
//' \itemize{
//'   \item \eqn{R} is an \eqn{n \times n} spatial correlation matrix (Matérn),
//'   \item \eqn{\Sigma} is a \eqn{q \times q} cross–covariance matrix across outcomes.
//' }
//'
//' @param Y \eqn{n \times q} data matrix of outcomes, with \eqn{n} spatial sites and \eqn{q} outcomes.
//' @param X Optional \eqn{n \times p} covariate matrix. If provided (\eqn{p > 0}),
//'   the model uses \eqn{Y|\beta, \Sigma \sim MN(X \beta, R_\theta, \Sigma)} and samples \eqn{\beta} by Gibbs.
//'   The prior is \eqn{\beta|\Sigma \sim MN(0_{p \times q}, 1e4 \cdot I_p, \Sigma)} by default.
//' @param coords \eqn{n \times d} matrix of spatial coordinates for the \eqn{n} sites.
//' @param custom_dag Field of index vectors defining the Vecchia approximation
//'   DAG structure for the sites. Use package \code{spiox} for building the DAG.
//' @param theta_start Numeric vector of latent GP hyperparameters
//'   (e.g. range, variance, smoothness, nugget). These are shared across all outcomes
//'   under the separable specification.
//' @param sampling Logical vector indicating which hyperparameters in \code{theta_start}
//'   are to be updated by MCMC.
//' @param Sigma_start Initial \eqn{q \times q} cross–covariance matrix \eqn{\Sigma}.
//' @param mcmc Integer, number of MCMC iterations (default 1000).
//' @param print_every Integer, print progress every this many iterations (default 100).
//' @param dag_opts Integer controlling Vecchia DAG modification:
//'   \itemize{
//'     \item \code{-1}: assume \code{coords} are gridded and DAG was built accordingly,
//'     \item \code{0}: use \code{custom_dag} as provided,
//'     \item \code{>0}: prune the DAG by up to this many neighbors to reduce parent set size.
//'   }
//' @param upd_Sigma Logical, whether to update the cross–covariance matrix \eqn{\Sigma} (default TRUE).
//' @param upd_theta Logical, whether to update GP hyperparameters \eqn{\theta} (default TRUE).
//' @param num_threads Integer, number of OpenMP threads to use (default 1).
//'
//' @return A list with elements:
//' \item{Sigma}{\eqn{q \times q \times mcmc} array of posterior samples of \eqn{\Sigma}.}
//' \item{theta}{Matrix of posterior samples of hyperparameters (rows = parameter, cols = iterations).}
//' \item{dag_cache}{DAG structure used by the Vecchia approximation (for reference).}
//'
//' @details
//' This function constructs a separable multivariate GP model and runs MCMC
//' updates for the spatial hyperparameters and the cross–covariance matrix.
//' Updates for \eqn{\Sigma} are performed via its inverse–Wishart full conditional.
//' Computation can be parallelized using OpenMP if available.
//'
//' @examples
//' \dontrun{
//'   # Example data
//'   n <- 50; q <- 2
//'   coords <- matrix(runif(n*2), n, 2)
//'   Y <- matrix(rnorm(n*q), n, q)
//'   theta_start <- c(1, 1, 0.5, 0.1) # (range, var, smoothness, nugget)
//'   Sigma_start <- diag(q)
//'   custom_dag <- some_dag_constructor(coords)
//'
//'   fit <- spseparable_response(Y, coords, custom_dag,
//'                               theta_start, sampling = c(1,1,1,1),
//'                               Sigma_start, mcmc = 200)
//'   str(fit)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List spseparable_response(const arma::mat& Y, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          
                          arma::vec theta_start,
                          arma::vec sampling,
                          const arma::mat& Sigma_start,
                          Rcpp::Nullable<arma::mat> X = R_NilValue,  // (X can be NULL)
                          
                          int mcmc = 1000,
                          int print_every = 100,
                          int dag_opts = 0,
                          bool upd_Sigma = true,
                          bool upd_theta = true,
                          int num_threads = 1){
  
  Rcpp::Rcout << "GP-separable coregionalization response model." << endl;
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  Separable spsep(Y, coords, Sigma_start, theta_start, sampling, custom_dag, dag_opts,
                  num_threads);
  
  // detect if covariates exists and turn them on with default priors
  bool haveX = X.isNotNull();
  arma::mat Xmat;
  if (haveX) {
    Xmat = Rcpp::as<arma::mat>(X);
    if (Xmat.n_cols == 0) haveX = false;
  }
  if (haveX) {
    if (Xmat.n_rows != Y.n_rows) Rcpp::stop("X must have nrow(X) == nrow(Y).");
    spsep.enable_covariates(Xmat, q);  // sets M0=0, V0_inv=(1e-4)I as default priors of beta
  }
  
  // storage
  arma::mat theta = arma::zeros(4, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  //for Beta - save in dim (p x q x mcmc)
  arma::cube Bsave;
  if (spsep.use_covariates){
    Bsave.set_size(Xmat.n_cols, q, mcmc);
  } 
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    if(upd_theta){
      spsep.upd_theta_metrop();
    }
    if(upd_Sigma){
      spsep.sample_iwishart();
    }
    
    // sample beta if there are covariates
    if (spsep.use_covariates){
      spsep.upd_beta_gibbs();
    }
    
    Sigma.slice(m) = spsep.Sigma_;
    theta.col(m) = spsep.theta_;
    
    // save for BETA
    if (spsep.use_covariates){
      Bsave.slice(m) = spsep.Beta_;
    }
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  // Adjusted for B (w. condition!)
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("dag_cache") = spsep.gp.dag_cache
  );
  
  if (spsep.use_covariates){
    res["Beta"] = Bsave;  
  }
  
  return res;
}

// [[Rcpp::export]]
double spseparable_logdens(const arma::mat& Y, 
                                const arma::mat& coords,
                                
                                const arma::field<arma::uvec>& custom_dag,
                                
                                arma::vec theta,
                                const arma::mat& Sigma){
  
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  Separable spsep(Y, coords, Sigma, theta, theta, custom_dag, 0, 1);
  
  return spsep.logdens(spsep.gp);
}

