#include <RcppArmadillo.h>
#include "daggp.h"
#include "ramadapt.h"

class Separable {
public:
  // A: q x q. theta: 4 x q with rows (phi, sigmasq, nu, tausq). gps: size q.
  Separable(const arma::mat& Y,
    const arma::mat& coords, const arma::mat& Sigma, 
      const arma::vec& theta, const arma::vec& sampling, 
      const arma::field<arma::uvec>& dag, 
      int dagopts, int nthreads)
    : Y_(Y), q(Sigma.n_rows), Sigma_(Sigma)
  {
    
    n = coords.n_rows;                         // initialize n_
    matern = true;
    
    theta_ = theta;
    n_options = theta_.n_cols;
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = sampling(0) != 0;
    sigmasq_sampling = sampling(1) != 0;
    nu_sampling = sampling(2) != 0;
    tausq_sampling = sampling(3) != 0;
    
    gp = DagGP(coords, theta, dag, 
               dagopts, true,  0, // with q blocks, make Ci
                nthreads);
    
    //daggp_options_alt = daggp_options;
    
    S0 = arma::eye(q,q);
    n0 = q+1;
    
    init_adapt();
  }
  
  std::size_t n_sites() const { return n; }
  std::size_t q_dim()   const { return q; }
  
  arma::mat Y_;
  std::size_t q, n;
  arma::mat Sigma_;
  
  int n0;
  arma::mat S0;
  
  DagGP gp;
  arma::vec theta_; // each column is one alternative value for theta
  unsigned int n_options;
  
  bool matern;
  
  bool phi_sampling, sigmasq_sampling, nu_sampling, tausq_sampling;
  // adaptive metropolis to update theta atoms
  arma::uvec which_theta_elem;
  
  // adaptive metropolis (conditional update) to update theta atoms
  // assume shared covariance functions and unknown parameters across variables
  arma::mat theta_unif_bounds;
  RAMAdapt theta_adapt;
  int theta_mcmc_counter;
  
  double logdens(const DagGP& gp) const {
    // u = vec(Y * A^{-T})
    arma::mat AinvT_ = arma::inv(arma::trimatu(arma::chol(Sigma_, "upper")));
    double logdetSigma = -2*arma::accu(log(AinvT_.diag()));
    arma::mat HW = gp.H * Y_ * AinvT_;
    double quad = arma::accu(arma::square(HW)); 
    
    const double c = -0.5 * double(n*q) * std::log(2.0 * M_PI);
    const double rowC = +0.5 * double(q) * gp.precision_logdeterminant; 
    const double colC = -0.5 * double(n) * logdetSigma;
    return c + rowC + colC - 0.5 * quad;
  }
  
  void init_adapt();
  void upd_theta_metrop();
  void sample_iwishart();
  
};

inline void Separable::init_adapt(){
  // adaptive metropolis
  theta_mcmc_counter = 0;
  which_theta_elem = arma::zeros<arma::uvec>(0);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  if(phi_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 0*oneuv);
  }
  if(sigmasq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 1*oneuv);
  }
  if(nu_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 2*oneuv);
  }
  if(tausq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({.3, 100}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({0.49, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  bounds_all.row(3) = arma::rowvec({1e-16, 100}); // tausq
  bounds_all = bounds_all.rows(which_theta_elem);
  
  //if(n_options == q){
  // conditional update
  theta_unif_bounds = bounds_all;
  int theta_par = which_theta_elem.n_elem;
  arma::mat theta_metrop_sd = 0.05 * arma::eye(theta_par, theta_par);
  theta_adapt = RAMAdapt(theta_par, theta_metrop_sd, 0.24);
  
  
}

inline void Separable::upd_theta_metrop() {
  arma::vec cur = theta_(which_theta_elem);     // subset to updated elements
  
  theta_adapt.count_proposal();
  Rcpp::RNGScope scope;
  arma::vec z = arma::randn(cur.n_elem);
  
  // propose on transformed scale, map back into [bounds]
  arma::vec alt = par_huvtransf_back(
    par_huvtransf_fwd(cur, theta_unif_bounds) + theta_adapt.paramsd * z,
    theta_unif_bounds
  );
  if (!alt.is_finite()) Rcpp::stop("theta out of bounds");
  
  arma::vec theta_alt = theta_;
  theta_alt(which_theta_elem) = alt;
  
  // proposal GP (do not mutate gp_ yet)
  DagGP gp_prop = gp;
  gp_prop.update_theta(theta_alt, true);
  
  const double curr_logdens = logdens(gp);
  const double prop_logdens = logdens(gp_prop);
  
  // priors (edit as needed: indices 0=phi,1=sig2,2=nu,3=tausq)
  double logpriors = 0.0;
  if (sigmasq_sampling)
    logpriors += invgamma_logdens(theta_alt(1), 2, 1) - invgamma_logdens(theta_(1), 2, 1);
  if (tausq_sampling)
    logpriors += expon_logdens(theta_alt(3), 25) - expon_logdens(theta_(3), 25);
  
  const double jac = calc_jacobian(alt, cur, theta_unif_bounds);
  const double logaccept = prop_logdens - curr_logdens + jac + logpriors;
  
  const bool accepted = do_I_accept(logaccept);
  if (accepted) {
    theta_ = theta_alt;
    std::swap(gp, gp_prop);
  }
  
  theta_adapt.update_ratios();
  theta_adapt.adapt(z, std::exp(std::min(0.0, logaccept)), theta_mcmc_counter);
  ++theta_mcmc_counter;
}

inline void Separable::sample_iwishart(){
  int n = Y_.n_rows;
  int q = Y_.n_cols;
  arma::mat HY = gp.H_times_A(Y_);
  arma::mat Smean = n * arma::cov(HY) + S0;
  arma::mat Q_mean_post = arma::inv_sympd(Smean);
  double df_post = n + n0;
  
  arma::mat Q = arma::wishrnd(Q_mean_post, df_post);
  arma::mat Si = arma::chol(Q, "lower");
  arma::mat S = arma::inv(arma::trimatl(Si));
  Sigma_ = S.t() * S;
}