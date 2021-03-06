// ==============================================================================
// author          :Ghislain Vieilledent, Jeanne Clement
// email           :ghislain.vieilledent@cirad.fr, jeanne.clement16@laposte.net
// web             :https://ecology.ghislainv.fr
// license         :GPLv3
// ==============================================================================

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
/* mat_select_lines*/
arma::mat mat_select_lines(arma::mat& X, arma::uvec& rowId) {
  int nrows = rowId.n_elem;
  int ncols = X.n_cols;
  arma::mat R(nrows, ncols);
  for(int i = 0; i < nrows; i++) {
    for(int j = 0; j < ncols; j++) {
      R(i, j) = X(rowId(i), j);
    }
  }
  return R;
}

/* ************************************************************ */
/* Gibbs sampler function */

// [[Rcpp::export]]
Rcpp::List Rcpp_jSDM_probit_traits_block(const int ngibbs, int nthin, int nburn, 
                                         arma::mat Y,
                                         arma::mat X,
                                         arma::mat D,
                                         arma::uvec Id_sp,
                                         arma::uvec Id_site,
                                         arma::mat sp_params_start,
                                         arma::mat V_sp_params,
                                         arma::vec mu_sp_params,
                                         arma::vec params_start,
                                         arma::mat V_params,
                                         arma::vec mu_params,
                                         arma::mat V_W,
                                         arma::mat W_start,
                                         arma::vec alpha_start,
                                         double Valpha_start,
                                         double shape,
                                         double rate,
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
  const int NL = W_start.n_cols; 
  const int NP = mu_sp_params.n_elem - NL;
  const int ND = mu_params.n_elem;
  const int NSITE = Y.n_rows;
  const int NSP = Y.n_cols;
  
  ///////////////////////////////////////////
  // Declaring new objects to store results //
  /* parameters */
  arma::Cube<double> sp_params; sp_params.zeros(NSAMP, NSP, NP+NL);
  arma::mat params; params.zeros(NSAMP, ND);
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
  
  // mat of species effects parameters and coefficients for latent variables (nl+np,nsp)
  arma::mat sp_params_run = sp_params_start;
  // vec of parameters (nd)
  arma::vec params_run = params_start;
  // alpha vec of sites effects (nsite)
  arma::vec alpha_run = alpha_start;
  double Valpha_run = Valpha_start;
  // w latent variables (nsite*nl)
  arma::mat W_run = W_start;
  // Z latent (nsite*nsp)
  arma::mat Z_run; Z_run.zeros(NSITE,NSP);
  // probit_theta 
  arma::mat probit_theta_run; probit_theta_run.zeros(NSITE,NSP);
  
  arma::mat M = arma::join_rows(X,W_start);
  
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
    
    for (int j=0; j<NSP; j++) {
      for (int i=0; i<NSITE; i++) {
        // Actualization
        if (Y(i,j) == 0) {
          Z_run(i,j) = rtnorm(s, R_NegInf, 0, probit_theta_run(i,j), 1);
        } else {
          Z_run(i,j) = rtnorm(s, 0, R_PosInf, probit_theta_run(i,j), 1);
        }
      }
    }
    
    /////////////////////////////////////////////
    // mat latent variable W: Gibbs algorithm //
    
    // Loop on sites
    for (int i=0; i<NSITE; i++) {
      // Di_site data corresponding to params for site i 
      arma::uvec rowId_i = find(Id_site == i);
      arma::mat Di_site= mat_select_lines(D, rowId_i);
      // beta 
      arma::mat beta_run = sp_params_run.submat(0,0,NP-1,NSP-1);
      // lambda
      arma::mat lambda_run = sp_params_run.submat(NP,0,NP+NL-1,NSP-1);
      // big_V
      arma::mat big_V = inv(inv(V_W)+lambda_run*lambda_run.t());
      
      // small_v
      arma::vec small_v = lambda_run*(Z_run.row(i)-X.row(i)*beta_run - (Di_site*params_run).t() - alpha_run(i)).t();
      
      // Draw in the posterior distribution
      arma::vec W_i = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
      W_run.row(i) = W_i.t();
    }
    
    M = arma::join_rows(X, W_run);
    
    ///////////////////////////////
    // vec alpha : Gibbs algorithm //
    
    // Loop on sites 
    double sum = 0.0;
    for (int i=0; i<NSITE; i++) {
      // Di_site data corresponding to params for site i 
      arma::uvec rowId_i = find(Id_site == i);
      arma::mat Di_site = mat_select_lines(D, rowId_i);
      
      // small_v
      double small_v = arma::sum(Z_run.row(i)-M.row(i)*sp_params_run - (Di_site*params_run).t());
      
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
    
    //////////////////////////////////
    // mat sp_params: Gibbs algorithm //
    
    // Loop on species
    for (int j=0; j<NSP; j++) {
      arma::uvec rowId_j = find(Id_sp == j);
      arma::mat Dj_sp = mat_select_lines(D, rowId_j);
      
      // small_v
      arma::vec small_v = inv(V_sp_params)*mu_sp_params + M.t()*(Z_run.col(j) - Dj_sp*params_run - alpha_run);
      // big_V
      arma::mat big_V = inv(inv(V_sp_params)+ M.t()*M);
      
      // Draw in the posterior distribution
      arma::vec sp_params_prop = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
      
      // constraints on lambda
      for (int l=0; l<NL; l++) {
        if (l > j) {
          sp_params_prop(NP+l) = 0;
        }
        if ((l==j) & (sp_params_prop(NP+l) < 0)) {
          sp_params_prop(NP+l) = sp_params_run(NP+l,j);
        }
      }
      sp_params_run.col(j) = sp_params_prop;
    }
    
    
    //////////////////////////////////
    // vec params: Gibbs algorithm //
    arma::vec small_v; small_v.zeros(ND);
    
    // Loop on species
    for (int j=0; j<NSP; j++) {
      arma::uvec rowId_j = find(Id_sp == j);
      arma::mat Dj_sp = mat_select_lines(D, rowId_j);
      
      // small_v
      small_v += Dj_sp.t()*(Z_run.col(j) - M*sp_params_run.col(j) - alpha_run);
    }
    small_v += inv(V_params)*mu_params;
    
    // big_V
    arma::mat big_V = inv(inv(V_params) +  D.t()*D);
    
    // Draw in the posterior distribution
    arma::vec params_run = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
    
    
    //////////////////////////////////////////////////
    //// Deviance
    
    // logLikelihood
    double logL = 0.0;
    for ( int i = 0; i < NSITE; i++ ) {
      for ( int j = 0; j < NSP; j++ ) {
        arma::uvec rowId_i = find(Id_site == i);
        arma::mat Di_site = mat_select_lines(D, rowId_i);
        arma::uvec rowId_j = find(Id_sp(rowId_i) == j);
        arma::mat D_ij= mat_select_lines(Di_site, rowId_j);
        
        
        // probit(theta_ij) = X_i*beta_j + W_i*lambda_j + D_ij*b + alpha_i 
        probit_theta_run(i,j) = arma::as_scalar(M.row(i)*sp_params_run.col(j) + D_ij*params_run + alpha_run(i));
        // link function probit is the inverse of N(0,1) repartition function 
        double theta = gsl_cdf_ugaussian_P(probit_theta_run(i,j));
        
        /* log Likelihood */
        logL += R::dbinom(Y(i,j), 1, theta, 1);
      } // loop on species
    } // loop on sites
    
    // Deviance
    double Deviance_run = -2 * logL;
    
    //////////////////////////////////////////////////
    // Output
    if (((g+1)>NBURN) && (((g+1)%(NTHIN))==0)) {
      int isamp=((g+1)-NBURN)/(NTHIN);
      for ( int j=0; j<NSP; j++ ) {
        sp_params.tube(isamp-1,j) = sp_params_run.col(j);
        for ( int i=0; i<NSITE; i++ ) {
          W.tube(isamp-1,i) = W_run.row(i);
          Z_latent(i,j) += Z_run(i,j) / NSAMP; // We compute the mean of NSAMP values
          probit_theta_pred(i,j) += probit_theta_run(i,j)/NSAMP;        
        }
      }
      params.row(isamp-1) = params_run.t();
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
    
    
    //////////////////////////////////////////////////
    // User interrupt
    R_CheckUserInterrupt(); // allow user interrupts 	    
    
  } // Gibbs sampler
  
  // Free memory
  gsl_rng_free(s);
  
  // Return results as a Rcpp::List
  Rcpp::List results = Rcpp::List::create(Rcpp::Named("sp_params") = sp_params,
                                          Rcpp::Named("params") = params,
                                          Rcpp::Named("W") = W,
                                          Rcpp::Named("alpha") = alpha,
                                          Rcpp::Named("Valpha") = Valpha,
                                          Rcpp::Named("Deviance") = Deviance,
                                          Rcpp::Named("Z_latent") = Z_latent,
                                          Rcpp::Named("probit_theta_pred") = probit_theta_pred );  
  return results;
  
} // end Rcpp_jSDM_probit_traits_block

// Test
/*** R
# ===================================================
# Data
# ===================================================
# 
# nsp<- 50
# nsite <- 500
# nl <- 2
# seed <- 1234
# set.seed(seed)
# 
# # Ecological process (suitability)
# x1 <- rnorm(nsite,0,1)
# X <- cbind(rep(1,nsite),x1)
# colnames(X) <- c("Int","x1")
# np <- ncol(X)
# W1 <- rnorm(nsite,0,1)
# W2 <- rnorm(nsite,0,1)
# W <- cbind(W1,W2)
# beta.target <- t(matrix(runif(nsp*np,-2,2), byrow=TRUE, nrow=nsp))
# l.zero <- 0
# l.diag <- runif(2,0,2)
# l.other <- runif(nsp*2-3,-2,2)
# lambda.target <- t(matrix(c(l.diag[1],l.zero,l.other[1],l.diag[2],l.other[-1]), byrow=T, nrow=nsp))
# sp_params.target <- rbind(beta.target,lambda.target)
# x2 <- x1^2
# D <- cbind(x1,x2)
# nd <- ncol(D)
# params.target <- runif(nd,-3,3)
# Valpha.target <- 2
# V <- 1
# alpha.target <- rnorm(nsite,0,sqrt(Valpha.target))
# traits <- rnorm(nsp,0,1)
# inter_param.target <- 2
# probit_theta = X %*% beta.target + W %*% lambda.target + as.vector(D %*% params.target) + inter_param.target*x1 %*% t(traits) + alpha.target
# e = matrix(rnorm(nsp*nsite,0,sqrt(V)),nsite,nsp)
# Z_true <- probit_theta + e
# visits <- matrix(1,nsite,nsp)
# Y = matrix (NA, nsite,nsp)
# for (i in 1:nsite){
#   for (j in 1:nsp){
#     if ( Z_true[i,j] > 0) {Y[i,j] <- 1}
#     else {Y[i,j] <- 0}
#   }
# }
# params.target <- c(params.target,inter_param.target)
# sp_params_start=matrix(0,np+nl,nsp)
# for (i in 1:nl){
#   sp_params_start[np+i,i] = 1
# }
# 
# data <- data.frame(site = rep(c(1:nsite),nsp)
#                    , species = rep(c(1:nsp),each=nsite),
#                    PA = c(Y),
#                    intercept = rep(1,nsp*nsite),
#                    x1 = rep(x1,nsp), x2 = rep(x2,nsp),
#                    x1.SLA = c(x1 %*% t(traits)))
# Id_sp <- rep(c(1:nsp),each=nsite)-1
# Id_site <- rep(c(1:nsite),nsp)-1
# # Call to C++ function
# # Iterations
# nsamp <- 5000
# nburn <- 15000
# nthin <- 5
# ngibbs <- nsamp+nburn
# mod <- Rcpp_jSDM_probit_traits_block(ngibbs=ngibbs, nthin=nthin, nburn=nburn,
#                                      Y=Y, X=X, D=cbind(data$x1,data$x2,data$x1.SLA),
#                                      Id_sp = Id_sp, Id_site=Id_site,sp_params_start=sp_params_start,
#                                      V_sp_params=diag(c(rep(1.0E6,np),rep(10,nl))), mu_sp_params = rep(0,np+nl),
#                                      params_start=rep(0,nd+1), V_params=diag(rep(10,nd+1)), mu_params = rep(0,nd+1),
#                                      W_start=matrix(0,nsite,nl), V_W=diag(rep(1,nl)),
#                                      alpha_start=rep(0,nsite), Valpha_start=1, shape=0.5, rate=0.0005,
#                                      seed=1234, verbose=1)
# 
# # ===================================================
# # Result analysis
# # ===================================================
# 
# # Parameters estimates
# ## probit_theta
# par(mfrow=c(1,1))
# plot(probit_theta,mod$probit_theta_pred)
# abline(a=0,b=1,col='red')
# ## Z
# plot(Z_true,mod$Z_latent)
# abline(a=0,b=1,col='red')
# ## alpha
# MCMC_alpha <- coda::mcmc(mod$alpha, start=nburn+1, end=ngibbs, thin=nthin)
# plot(alpha.target,summary(MCMC_alpha)[[1]][,"Mean"], ylab ="alpha.estimated")
# abline(a=0,b=1,col='red')
# ## Valpha
# MCMC_Valpha <- coda::mcmc(mod$Valpha, start=nburn+1, end=ngibbs, thin=nthin)
# summary(MCMC_Valpha)
# par(mfrow=c(1,2))
# coda::traceplot(MCMC_Valpha)
# coda::densplot(MCMC_Valpha)
# abline(v=Valpha.target,col='red')
# 
# ## beta_j
# par(mfrow=c(np,2))
# for (j in 1:4) {
#   for (p in 1:np) {
#     MCMC.betaj <- coda::mcmc(mod$sp_params[,j,1:np], start=nburn+1, end=ngibbs, thin=nthin)
#     summary(MCMC.betaj)
#     coda::traceplot(MCMC.betaj[,p])
#     coda::densplot(MCMC.betaj[,p], main = paste0("beta",p,j))
#     abline(v=beta.target[p,j],col='red')
#   }
# }
# ## lambda_j
# par(mfrow=c(nl*2,2))
# for (j in 1:4) {
#   for (l in 1:nl) {
#     MCMC.lambdaj <- coda::mcmc(mod$sp_params[,j,(np+1):(nl+np)], start=nburn+1, end=ngibbs, thin=nthin)
#     summary(MCMC.lambdaj)
#     coda::traceplot(MCMC.lambdaj[,l])
#     coda::densplot(MCMC.lambdaj[,l],main = paste0("lambda",l,j))
#     abline(v=lambda.target[l,j],col='red')
#   }
# }
# 
# ## params
# par(mfrow=c(nd+1,2))
# for (d in 1:(nd+1)) {
#   MCMC.params <- coda::mcmc(mod$params, start=nburn+1, end=ngibbs, thin=nthin)
#   summary(MCMC.params)
#   coda::traceplot(MCMC.params[,d])
#   coda::densplot(MCMC.params[,d],main = paste0("b_",d))
#   abline(v=params.target[d],col='red')
# }
# 
# ## W latent variables
# par(mfrow=c(1,1))
# MCMC.vl1 <- coda::mcmc(mod$W[,,1], start=nburn+1, end=ngibbs, thin=nthin)
# MCMC.vl2 <- coda::mcmc(mod$W[,,2], start=nburn+1, end=ngibbs, thin=nthin)
# plot(W[,1],summary(MCMC.vl1)[[1]][,"Mean"])
# abline(a=0,b=1,col='red')
# plot(W[,2],summary(MCMC.vl2)[[1]][,"Mean"])
# abline(a=0,b=1,col='red')
# ## Deviance
# mean(mod$Deviance)
*/
