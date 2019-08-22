#include <RcppArmadillo.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <cmath>
#include "Rcpp_jSDM_useful.h" 

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

using namespace arma;
using namespace std;

/* ************************************************************ */
/* Gibbs sampler function */

// [[Rcpp::export]]
Rcpp::List Rcpp_hSDM_binomial_probit(const int ngibbs, int nthin, int nburn, // Number of iterations, burning and samples
                                         arma::uvec Y, // Number of successes (presences)
                                         arma::uvec T, // Number of trials
                                         arma::mat X, // Suitability covariates
                                         arma::vec beta_start,
                                         arma::vec mubeta,
                                         arma::mat Vbeta,
                                         const int seed,
                                         const int verbose) {
  
  ////////////////////////////////////////////////////////////////////////////////
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Defining and initializing objects
  
  ////////////////////////////////////////
  // Initialize random number generator //
  gsl_rng *s = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(s, seed);
  
  ///////////////////////////
  // Redefining constants //
  const int NGIBBS = ngibbs;
  const int NTHIN = nthin;
  const int NBURN = nburn;
  const int NSAMP = (NGIBBS-NBURN)/NTHIN;
  const int NSITE = X.n_rows;
  const int NP = X.n_cols;
  
  ///////////////////////////////////////////
  // Declaring new objects to store results //
  /* Parameters */
  arma::mat beta; beta.zeros(NSAMP, NP);
  
  /* Latent variable */
  arma::vec probit_theta_pred; probit_theta_pred.zeros(NSITE);
  arma::vec Z_latent; Z_latent.zeros(NSITE);
  /* Deviance */
  arma::vec Deviance; Deviance.zeros(NSAMP);
  
  /////////////////////////////////////
  // Initializing running parameters //
  
  // vector of species effects parameters (np,1)
  arma::vec beta_run = beta_start;
  // Z latent (nsite*nsp)
  arma::vec Z_run; Z_run.zeros(NSITE);
  // probit_theta_i = X_i*beta
  arma::vec probit_theta_run ;probit_theta_run.zeros(NSITE);
  
  ////////////
  // Message//
  Rprintf("\nRunning the Gibbs sampler. It may be long, please keep cool :)\n\n");
  R_FlushConsole();
  
  ///////////////////////////////////////////////////////////////////////////////////////
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Gibbs sampler 
  
  for (int g=0; g < NGIBBS; g++) {
    
    ////////////////////////////////////////////////
    // latent variable Z // 
    
    for (int i=0; i<NSITE; i++) {
      
      // Mean of the prior
      probit_theta_run(i) = arma::as_scalar(X.row(i)*beta_run);
      
      // Actualization
      if ( Y(i) == 1) {
        Z_run(i) = rtnorm(s,0,R_PosInf,probit_theta_run(i), 1);
      }
      
      if ( Y(i) == 0) {
        Z_run(i) = rtnorm(s,R_NegInf,0,probit_theta_run(i), 1);
      }
    }
  
  
  //////////////////////////////////
  // vec beta : Gibbs algorithm //
  
    // small_v
  arma::vec small_v = arma::inv(Vbeta)*mubeta + X.t()*Z_run;
  // big_V
  arma::mat big_V = arma::inv(arma::inv(Vbeta)+ X.t()*X);
  arma::mat chol_Var = chol_decomp(big_V);
  // Draw in the posterior distribution
  beta_run = arma_mvgauss(s, big_V*small_v, chol_Var);
  
  
  
  //////////////////////////////////////////////////
  //// Deviance
  
  // logLikelihood
  double logL = 0.0;
  for ( int i = 0; i < NSITE; i++ ) {
    // probit(theta) = Z_run
    // link function probit = inverse of N(0,1) repartition function 
    double theta = gsl_cdf_ugaussian_P(arma::as_scalar(X.row(i)*beta_run));
    
    /* log Likelihood */
    logL += R::dbinom(Y(i), T(i), theta, 1);
  } // loop on sites
  
  // Deviance
  double Deviance_run = -2 * logL;
  
  //////////////////////////////////////////////////
  // Output
  if (((g+1)>NBURN) && (((g+1)%(NTHIN))==0)) {
    int isamp=((g+1)-NBURN)/(NTHIN);
    for ( int i=0; i<NSITE; i++ ) {
      Z_latent(i) += Z_run(i) / NSAMP; // We compute the mean of NSAMP values
      probit_theta_pred(i) += probit_theta_run(i)/NSAMP;        
    }
    beta.row(isamp-1) = beta_run.t();
    Deviance(isamp-1) = Deviance_run;
  }
  
  //////////////////////////////////////////////////
  // Progress bar
  double Perc=100*(g+1)/(NGIBBS);
  if (((g+1)%(NGIBBS/100))==0 && (verbose==1)) {  
    Rprintf("*");
    R_FlushConsole();
    //R_ProcessEvents(); for windows
    if (((g+1)%(NGIBBS/10))==0) {
      Rprintf(":%.1f%% \n",Perc);
      R_FlushConsole();
      //R_ProcessEvents(); for windows
    }
  } 
  
  
  //////////////////////////////////////////////////
  // User interrupt
  R_CheckUserInterrupt(); // allow user interrupts 	    
  
} // Gibbs sampler

// Free memory
gsl_rng_free(s);

// Return results as a Rcpp::List
Rcpp::List results = Rcpp::List::create(Rcpp::Named("beta") = beta,
                                        Rcpp::Named("Deviance") = Deviance,
                                        Rcpp::Named("Z_latent") = Z_latent,
                                        Rcpp::Named("probit_theta_pred") = probit_theta_pred);  
return results;

} // end Rcpp_hSDM_binomial_probit
// Test
/*** R
#===================================================
# Data
#===================================================
# 
# nsite <- 500
# seed <- 123
# set.seed(seed)
# 
# # Ecological process (suitability)
# x1 <- rnorm(nsite,0,1)
# x2 <- rnorm(nsite,0,1)
# x3 <- rnorm(nsite,0,1)
# X <- cbind(rep(1,nsite),x1,x2,x3)
# np <- ncol(X)
# beta.target <- runif(np,-2,2)
# probit_theta = X %*% beta.target 
# e = rnorm(nsite,0,1)
# Z_true <- probit_theta + e
# visits <- rep(1,nsite)
# #comment on fait pour prendre en compte le nombre de visites ?
# Y = rep (NA, nsite)
# for (i in 1:nsite){
#     if ( Z_true[i] > 0) {Y[i] <- 1}
#     else {Y[i] <- 0}
#   }
# 
# # Call to C++ function
# # Iterations
# nsamp <- 5000
# nburn <- 5000
# nthin <- 5
# ngibbs <- nsamp+nburn
# mod <- Rcpp_hSDM_binomial_probit(ngibbs=ngibbs, nthin=nthin, nburn=nburn,
#                                 Y=Y,T=visits,X=X,
#                                 beta_start=rep(0,np), mubeta= rep(0,np), Vbeta=diag(rep(1.0E6,np)),
#                                 seed=123, verbose=1)
# 
# # ===================================================
# # Result analysis
# # ===================================================
# 
# # Parameter estimates
# ## probit_theta
# par(mfrow=c(1,1))
# plot(probit_theta,mod$probit_theta_pred)
# abline(a=0,b=1,col='red')
# ## Z
# plot(Z_true,mod$Z_latent)
# abline(a=0,b=1,col='red')
# 
# ## beta
# MCMC.beta <- coda::mcmc(mod$beta, start=nburn+1, end=ngibbs, thin=nthin)
# summary(MCMC.beta)
# par(mfrow=c(np,2))
#   for (p in 1:np) {
#     coda::traceplot(MCMC.beta[,p])
#     coda::densplot(MCMC.beta[,p], main = paste0("beta",p))
#     abline(v=beta.target[p],col='red')
#   }
# 
# ## Deviance
# mean(mod$Deviance)
*/
