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
/* Gibbs sampler function */

// [[Rcpp::export]]
Rcpp::List Rcpp_jSDM_probit_block(const int ngibbs, int nthin, int nburn, 
                                  arma::umat Y, 
                                  arma::umat T, 
                                  arma::mat X,
                                  arma::mat D,
                                  arma::mat C,
                                  arma::mat sp_params_start,
                                  arma::mat V_sp_params,
                                  arma::vec mu_sp_params,
                                  arma::vec params_start,
                                  arma::mat V_params,
                                  arma::vec mu_params,
                                  arma::mat VW,
                                  arma::mat W_start,
                                  arma::vec alpha_start,
                                  arma::mat Vb_start,
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
  const int NSITE = Y.n_rows;
  const int NP = X.n_cols;
  const int ND = D.n_cols;
  const int NSP = Y.n_cols;
  const int NL = W_start.n_cols; 
  const int NKNOTS = C.n_cols;
  
  
  ///////////////////////////////////////////
  // Declaring new objects to store results //
  /* Parameters */
  arma::Cube<double> sp_params; sp_params.zeros(NSAMP, NSP, NP+NL);
  arma::mat params; params.zeros(NSAMP, ND);
  arma::Cube<double> W; W.zeros(NSAMP, NSITE, NL);
  arma::mat alpha; alpha.zeros(NSAMP, NSITE);
  arma::mat b; b.zeros(NSAMP, NKNOTS);
  //arma::vec Vb; Vb.zeros(NSAMP);
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
  arma::vec b_run; b_run.zeros(NKNOTS);
  arma::mat Vb_run = Vb_start;
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
    
    
    //////////////////////////////////
    // mat sp_params: Gibbs algorithm //
    
    // Loop on species
    for (int j=0; j<NSP; j++) {
      // small_v
      arma::vec small_v = inv( V_sp_params)* mu_sp_params + data.t()*(Z_run.col(j) - D*params_run - alpha_run);
      // big_V
      arma::mat big_V = inv(inv(V_sp_params)+data.t()*data);
      
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
    
    
    /////////////////////////////////////////////
    // mat latent variable W: Gibbs algorithm //
    
    // Loop on sites
    for (int i=0; i<NSITE; i++) {
      arma::mat beta_run = sp_params_run.submat(0,0,NP-1,NSP-1);
      arma::mat lambda_run = sp_params_run.submat(NP,0,NP+NL-1,NSP-1);
      // big_V
      arma::mat big_V = inv(inv(VW)+lambda_run*lambda_run.t());
      
      // small_v
      arma::vec small_v =lambda_run*(Z_run.row(i)-X.row(i)*beta_run - alpha_run(i)- arma::as_scalar(D.row(i)*params_run)).t();
      
      // Draw in the posterior distribution
      arma::vec W_i = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
      W_run.row(i) = W_i.t();
    }
    
    data = arma::join_rows(X, W_run);
    
    ///////////////////////////////
    // vec alpha : Gibbs algorithm //
    
    // 2D spline coefficients b_k 
    // Loop on sites 
    arma::vec small_v; small_v.zeros(NKNOTS);
    for (int j=0; j<NSP; j++) {
      // small_v
      small_v += C.t()*(Z_run.col(j) - data*sp_params_run.col(j) - D*params_run);
    }
    // big_V 
    arma::mat big_V = inv(inv(Vb_run) + C.t()*C*NSP);
    
    // Draw in the posterior distribution
    b_run = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
    
    double sum = arma::as_scalar(b_run.t()*b_run);
    // Centering b
    b_run = b_run - arma::sum(b_run)/NKNOTS;
    
    alpha_run = C * b_run;
    
    ////////////////////////////////////////////////
    // Vb
    // double shape_posterior = shape + 0.5*NKNOTS;
    // double rate_posterior = rate + 0.5*sum;
    // 
    // Vb_run = rate_posterior/gsl_ran_gamma_mt(s, shape_posterior, 1.0);
    
    //////////////////////////////////
    // vec params: Gibbs algorithm //
    arma::vec small_v_params; small_v_params.zeros(ND);
    
    // Loop on species
    for (int j=0; j<NSP; j++) {
      
      // small_v_params
      small_v_params += D.t()*(Z_run.col(j) - data*sp_params_run.col(j) - alpha_run);
    }
    small_v_params += inv(V_params)*mu_params;
    
    // big_V
    arma::mat big_V_params = inv(inv(V_params) +  D.t()*D*NSP);
    
    // Draw in the posterior distribution
    params_run = arma_mvgauss(s, big_V_params*small_v_params, chol_decomp(big_V_params));
    
    
    
    //////////////////////////////////////////////////
    //// Deviance
    
    // logLikelihood
    double logL = 0.0;
    for ( int i = 0; i < NSITE; i++ ) {
      for ( int j = 0; j < NSP; j++ ) {
        // probit(theta_ij) = X_i*beta_j + W_i*lambda_j + alpha_i 
        probit_theta_run(i,j) = arma::as_scalar(data.row(i)*sp_params_run.col(j) +  D.row(i)*params_run + alpha_run(i));
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
        sp_params.tube(isamp-1,j) = sp_params_run.col(j);
        for ( int i=0; i<NSITE; i++ ) {
          W.tube(isamp-1,i) = W_run.row(i);
          Z_latent(i,j) += Z_run(i,j) / NSAMP; // We compute the mean of NSAMP values
          probit_theta_pred(i,j) += probit_theta_run(i,j)/NSAMP;        
        }
      }
      alpha.row(isamp-1) = alpha_run.t();
      params.row(isamp-1) = params_run.t();
      b.row(isamp-1) = b_run.t();
      //Vb(isamp-1) = Vb_run;
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
                                          Rcpp::Named("b") = b,
                                          Rcpp::Named("Deviance") = Deviance,
                                          Rcpp::Named("Z_latent") = Z_latent,
                                          Rcpp::Named("probit_theta_pred") = probit_theta_pred);  
  return results;
  
} // end Rcpp_jSDM_probit_block

// Test
/*** R
# ===================================================
# Data
# ===================================================
library(raster) 
library(rdist)

# Function to draw in a multivariate normal 
rmvn <- function(n, mu=0, V=matrix(1), seed=1234) { 
  p <- length(mu) 
  if (any(is.na(match(dim(V), p)))) {
    stop("Dimension problem!") 
  } 
  D <- chol(V) 
  set.seed(seed) 
  t(matrix(rnorm(n*p),ncol=p)%*%D+rep(mu,rep(n,p))) 
}

data(altitude,package="hSDM")
# Number of observation sites 
nsite <- 500 
sites <- c(1:nsite)
nsp <- 50
nl <- 2 
# Set seed for repeatability 
seed <- 1234
set.seed(seed)
# Import covariates as a raster
alt.orig <- rasterFromXYZ(altitude) 
extent(alt.orig) <- c(0,50,0,50) 

# Define the knots locations, 50 knots regularly spaced
knotlocs <- cbind(rep(c(5,15,25,35,45),5),rep(c(5,15,25,35,45),each=5))
nknots <- nrow(knotlocs)

# Scaled coordinates of the sites
xCoord <- runif(nsite,0,50) 
yCoord <- runif(nsite,0,50) 
D = cbind(scale(xCoord),scale(yCoord))
nd <- ncol(D)
# params 
params.target <- runif(nd,-1,1)

plot(alt.orig) 
points(knotlocs, pch=3)
points(cbind(xCoord, yCoord), pch=2, col='red')

# Define the omega and Z matrices for the random effects
omega.all <- (pdist(knotlocs)/100)^3
svd.omega.all <- svd(omega.all)
sqrt.omega.all <- t(svd.omega.all$v %*% (t(svd.omega.all$u)*sqrt(svd.omega.all$d)))
C.k <- (cdist(cbind(xCoord, yCoord), knotlocs)/100)^3
C <- t(solve(sqrt.omega.all, t(C.k)))
# Center and scale altitudinal data
alt <- scale(alt.orig,center=TRUE,scale=TRUE) 
coords <- coordinates(alt)
plot(alt)
# Rasters must be projected to correctly compute the neighborhood
ncells <- ncell(alt)
library(sp) 
sites.sp <- SpatialPoints(coords=cbind(xCoord,yCoord)) 
cells <- extract(alt,sites.sp,cell=TRUE)[,1]
# Extract altitude data for sites 
alt.sites <- extract(alt,sites.sp) 
# Ecological process (suitability)
X <- cbind(rep(1,nsite),alt.sites)
colnames(X) <- c("Int","x1")
np <- ncol(X)
W <- cbind(rnorm(nsite,0,1),rnorm(nsite,0,1))
data = cbind (X,W)
beta.target <- t(matrix(runif(nsp*np,-1,1), byrow=TRUE, nrow=nsp))
l.zero <- 0
l.diag <- runif(2,0,1)
l.other <- runif(nsp*2-3,-1,1)
lambda.target <- t(matrix(c(l.diag[1],l.zero,l.other[1],l.diag[2],l.other[-1]), byrow=T, nrow=nsp))
sp_params.target <- rbind(beta.target,lambda.target)
Vb.target <- diag(rep(10,nknots))
b.target <- rmvn(1,mu=rep(0,nknots),V=Vb.target)
V <- 1
alpha.target <- C %*% b.target
probit_theta <- X %*% beta.target + W %*% lambda.target  + as.vector(alpha.target + D %*% params.target)
e <- matrix(rnorm(nsp*nsite,0,sqrt(V)),nsite,nsp)
Z_true <- probit_theta + e
visits <- matrix(1,nsite,nsp)
Y = matrix (NA, nsite,nsp)
for (i in 1:nsite){
  for (j in 1:nsp){
    if ( Z_true[i,j] > 0) {Y[i,j] <- 1}
    else {Y[i,j] <- 0}
  }
}

sp_params_start=matrix(0,np+nl,nsp)
for (i in 1:nl){
  sp_params_start[np+i,i] = 1
}

# Call to C++ function
# Iterations
nsamp <- 10000
nburn <- 140000
nthin <- 10
ngibbs <- nsamp+nburn
mod <- Rcpp_jSDM_probit_block(ngibbs=ngibbs, nthin=nthin, nburn=nburn,
                              Y=Y,T=visits,X=X,D=D,C=C,
                              sp_params_start=sp_params_start,
                              V_sp_params=diag(c(rep(100,,np),rep(10,nl))), mu_sp_params = rep(0,np+nl),
                              params_start = rep(0,nd), mu_params = rep(0,nd), V_params=diag(rep(100,nd)),
                              W_start=matrix(0,nsite,nl), VW=diag(rep(1,nl)),
                              alpha_start=rep(0,nsite), Vb_start=diag(rep(1,nknots)),
                              seed=123, verbose=1)

# ===================================================
# Result analysis
# ===================================================

# Parameter estimates
## probit_theta
par(mfrow=c(1,1))
plot(probit_theta,mod$probit_theta_pred)
abline(a=0,b=1,col='red')
## Z
plot(Z_true,mod$Z_latent)
abline(a=0,b=1,col='red')
## alpha
MCMC_alpha <- coda::mcmc(mod$alpha, start=nburn+1, end=ngibbs, thin=nthin)
plot(alpha.target,summary(MCMC_alpha)[[1]][,"Mean"], ylab ="alpha.estimated")
abline(a=0,b=1,col='red')
## b 2D_spline 
MCMC_b <- coda::mcmc(mod$b, start=nburn+1, end=ngibbs, thin=nthin)
plot(b.target,summary(MCMC_b)[[1]][,"Mean"], ylab ="b.estimated")
abline(a=0,b=1,col='red')

## b_k
par(mfrow=c(3,2))
for (k in 1:12) {
  MCMC.b <- coda::mcmc(mod$b, start=nburn+1, end=ngibbs, thin=nthin)
  summary(MCMC.b)
  coda::traceplot(MCMC.b[,k])
  coda::densplot(MCMC.b[,k], main = paste0("b_",k))
  abline(v=b.target[k],col='red')
}

## params
par(mfrow=c(2,3))
for (d in 1:nd) {
  MCMC.params <- coda::mcmc(mod$params, start=nburn+1, end=ngibbs, thin=nthin)
  summary(MCMC.params)
  coda::traceplot(MCMC.params[,d])
  coda::densplot(MCMC.params[,d], main = paste0("param_",d))
  abline(v=params.target[d],col='red')
}

## beta_j
par(mfrow=c(np,2))
for (j in 1:4) {
  for (p in 1:np) {
    MCMC.betaj <- coda::mcmc(mod$sp_params[,j,1:np], start=nburn+1, end=ngibbs, thin=nthin)
    summary(MCMC.betaj)
    coda::traceplot(MCMC.betaj[,p])
    coda::densplot(MCMC.betaj[,p], main = paste0("beta",p,j))
    abline(v=beta.target[p,j],col='red')
  }
}
## lambda_j
par(mfrow=c(nl*2,2))
for (j in 1:4) {
  for (l in 1:nl) {
    MCMC.lambdaj <- coda::mcmc(mod$sp_params[,j,(np+1):(nl+np)], start=nburn+1, end=ngibbs, thin=nthin)
    summary(MCMC.lambdaj)
    coda::traceplot(MCMC.lambdaj[,l])
    coda::densplot(MCMC.lambdaj[,l],main = paste0("lambda",l,j))
    abline(v=lambda.target[l,j],col='red')
  }
}

## W latent variables
par(mfrow=c(1,1))
MCMC.vl1 <- coda::mcmc(mod$W[,,1], start=nburn+1, end=ngibbs, thin=nthin)
MCMC.vl2 <- coda::mcmc(mod$W[,,2], start=nburn+1, end=ngibbs, thin=nthin)
plot(W[,1],summary(MCMC.vl1)[[1]][,"Mean"])
abline(a=0,b=1,col='red')
plot(W[,2],summary(MCMC.vl2)[[1]][,"Mean"])
abline(a=0,b=1,col='red')
## Deviance
mean(mod$Deviance)
*/
