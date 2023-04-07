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
Rcpp::List Rcpp_jSDM_probit_block(const int ngibbs, int nthin, int nburn, 
                                  arma::umat Y, 
                                  arma::umat T, 
                                  arma::mat X,
                                  arma::mat param_start,
                                  arma::mat Vparam,
                                  arma::vec muparam,
                                  arma::mat VW,
                                  arma::mat W_start,
                                  arma::vec alpha_start,
                                  double Valpha_start,
                                  double shape,
                                  double rate,
                                  const int seed,
                                  const int verbose) {
  
  ///////////////////////////////////////
  // Defining and initializing objects//
  
  // Initialize random number generator 
  gsl_rng *s = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(s, seed);
  
  // Redefining constants 
  const int NGIBBS = ngibbs;
  const int NTHIN = nthin;
  const int NBURN = nburn;
  const int NSAMP = (NGIBBS-NBURN)/NTHIN;
  const int NSITE = Y.n_rows;
  const int NP = X.n_cols;
  const int NSP = Y.n_cols;
  const int NL = W_start.n_cols; 
  
  /////////////////////////////////////////////
  // Declaring new objects to store results //
  /* Parameters */
  arma::Cube<double> param; param.zeros(NSAMP, NSP, NP+NL);
  arma::Cube<double> W; W.zeros(NSAMP, NSITE, NL);
  arma::mat alpha; alpha.zeros(NSAMP, NSITE);
  arma::vec Valpha; Valpha.zeros(NSAMP);
  /* Latent variable */
  arma::mat probit_theta_pred; probit_theta_pred.zeros(NSITE, NSP);
  arma::mat Z_latent; Z_latent.zeros(NSITE, NSP);
  /* Deviance */
  arma::vec Deviance; Deviance.zeros(NSAMP);
  
  /////////////////////////////////////
  // Initializing running parameters //
  
  //  mat of species effects parameters and coefficients for latent variables (nl+np,nsp)
  arma::mat param_run = param_start;
  // alpha vec of sites effects (nsite)
  arma::vec alpha_run = alpha_start;
  double Valpha_run = Valpha_start;
  // w latent variables (nsite*nl)
  arma::mat W_run = W_start;
  // Z latent (nsite*nsp)
  arma::mat Z_run; Z_run.zeros(NSITE,NSP);
  // probit_theta_ij = X_i*beta_j + W_i*lambda_j + alpha_i
  arma::mat probit_theta_run; probit_theta_run.zeros(NSITE,NSP);
  // data 
  arma::mat data = arma::join_rows(X,W_run);
  
  ////////////
  // Message//
  Rprintf("\nRunning the Gibbs sampler. It may be long, please keep cool :)\n\n");
  R_FlushConsole();
  
  ///////////////////////////////////////////////////////////////////////////////////////
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Gibbs sampler 
  
  for (int g=0; g < NGIBBS; g++) {
    
    ////////////////////////
    // latent variable Z // 
    
    for (int j=0; j<NSP; j++) {
      for (int i=0; i<NSITE; i++) {
        // Actualization
        if (Y(i,j) == 0) {
          Z_run(i,j) = rtnorm(s, R_NegInf, 0, probit_theta_run(i,j), 1);
        } else {
          Z_run(i,j) = rtnorm(s, 0, R_PosInf, probit_theta_run(i,j), 1);
        }
      } // loop on sites
    }// loop on species
    
    //////////////////////////////////
    // mat param: Gibbs algorithm //
    
    // Loop on species
    for (int j=0; j<NSP; j++) {
      // small_v
      arma::vec small_v = inv(Vparam)*muparam + data.t()*(Z_run.col(j) - alpha_run);
      // big_V
      arma::mat big_V = inv(inv(Vparam)+data.t()*data);
      // Draw in the posterior distribution
      arma::vec param_prop = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
      
      // constraints on lambda
      for (int l=0; l<NL; l++) {
        if (l > j) {
          param_prop(NP+l) = 0;
        }
        if ((l==j) & (param_prop(NP+l) < 0)) {
          param_prop(NP+l) = param_run(NP+l,j);
        }
      }
      param_run.col(j) = param_prop;
    }// loop on species
    
    
    /////////////////////////////////////////////
    // mat latent variable W: Gibbs algorithm //
    
    // Loop on sites
    for (int i=0; i<NSITE; i++) {
      arma::mat beta_run = param_run.submat(0,0,NP-1,NSP-1);
      arma::mat lambda_run = param_run.submat(NP,0,NP+NL-1,NSP-1);
      // big_V
      arma::mat big_V = inv(inv(VW)+lambda_run*lambda_run.t());
      
      // small_v
      arma::vec small_v =lambda_run*(Z_run.row(i)-X.row(i)*beta_run-alpha_run(i)).t();
      
      // Draw in the posterior distribution
      arma::vec W_i = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
      W_run.row(i) = W_i.t();
    }
    
    data = arma::join_rows(X, W_run);
    
    //////////////////////////////////
    // vec alpha : Gibbs algorithm //
    
    // Loop on sites 
    double sum = 0.0;
    for (int i=0; i<NSITE; i++) {
      // small_v
      double small_v = arma::sum(Z_run.row(i)-data.row(i)*param_run);
      // big_V
      double big_V = 1/(1/Valpha_run + NSP);
      
      // Draw in the posterior distribution
      alpha_run(i) = big_V*small_v + gsl_ran_gaussian_ziggurat(s, std::sqrt(big_V));
      sum += alpha_run(i)*alpha_run(i);
    }
    
    ////////////////////////////////////////////////
    // Valpha
    double shape_posterior = shape + 0.5*NSITE;
    double rate_posterior = rate + 0.5*sum;
    
    Valpha_run = rate_posterior/gsl_ran_gamma_mt(s, shape_posterior, 1.0);
    
    //////////////
    // Deviance //
    
    // logLikelihood
    double logL = 0.0;
    for ( int i = 0; i < NSITE; i++ ) {
      for ( int j = 0; j < NSP; j++ ) {
        // probit(theta_ij) = X_i*beta_j + W_i*lambda_j + alpha_i 
        probit_theta_run(i,j) = arma::as_scalar(data.row(i)*param_run.col(j) + alpha_run(i));
        // link function probit is the inverse of N(0,1) repartition function 
        double theta = gsl_cdf_ugaussian_P(probit_theta_run(i,j));
        /* log Likelihood */
        logL += R::dbinom(Y(i,j), T(i,j), theta, 1);
      } // loop on species
    } // loop on sites
    
    // Deviance
    double Deviance_run = -2 * logL;
    
    //////////////////////////////////////////////////
    // Output
    if (((g+1)>NBURN) && (((g+1)%(NTHIN))==0)) {
      int isamp=((g+1)-NBURN)/(NTHIN);
      for ( int j=0; j<NSP; j++ ) {
        param.tube(isamp-1,j) = param_run.col(j);
        for ( int i=0; i<NSITE; i++ ) {
          W.tube(isamp-1,i) = W_run.row(i);
          Z_latent(i,j) += Z_run(i,j) / NSAMP; // We compute the mean of NSAMP values
          probit_theta_pred(i,j) += probit_theta_run(i,j)/NSAMP;        
        }
      }
      alpha.row(isamp-1) = alpha_run.t();
      Valpha(isamp-1) = Valpha_run;
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
    
    ///////////////////
    // User interrupt //
    R_CheckUserInterrupt(); // allow user interrupts 	    
    
  } // Gibbs sampler
  
  // Free memory
  gsl_rng_free(s);
  
  // Return results as a Rcpp::List
  Rcpp::List results = Rcpp::List::create(Rcpp::Named("param") = param,
                                          Rcpp::Named("W") = W,
                                          Rcpp::Named("alpha") = alpha,
                                          Rcpp::Named("Valpha") = Valpha,
                                          Rcpp::Named("Deviance") = Deviance,
                                          Rcpp::Named("Z_latent") = Z_latent,
                                          Rcpp::Named("probit_theta_pred") = probit_theta_pred);  
  return results;
  
} // end Rcpp_jSDM_probit_block