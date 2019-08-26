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
/* vec_select_elems*/
arma::vec vec_select_elems(arma::vec& X, arma::uvec& rowId) {
  int n_elems = rowId.n_elem;
  arma::vec R(n_elems);
  for(int i = 0; i < n_elems; i++) {
    R(i) = X(rowId(i));
  }
  return R;
}


/* ************************************************************ */
/* Gibbs sampler function */

// [[Rcpp::export]]
Rcpp::List Rcpp_jSDM_probit_block(const int ngibbs, int nthin, int nburn, 
                                  arma::umat Y, 
                                  arma::mat X,
                                  arma::uvec cells,
                                  arma::mat X_pred,
                                  arma::uvec cells_pred,
                                  arma::vec n_neighbors,
                                  arma::mat mat_neighbors,
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
  const int NSP = Y.n_cols;
  const int NL = W_start.n_cols; 
  const int NCELL = X_pred.n_rows;
  arma::uvec visited_cells = arma::unique(cells);
  const int NVISCELL = visited_cells.n_elem;
  const int NPRED = X_pred.n_rows;
  
  /* Visited cell or not */
  arma::vec viscell; viscell.zeros(NCELL);
  for (int n=0; n<NVISCELL; n++) {
    viscell(visited_cells[n])++;
  }
  
  ///////////////////////////////////////////
  // Declaring new objects to store results //
  /* Parameters */
  arma::Cube<double> param; param.zeros(NSAMP, NSP, NP+NL);
  arma::Cube<double> W; W.zeros(NSAMP, NCELL, NL);
  arma::mat alpha; alpha.zeros(NSAMP, NCELL);
  arma::vec Valpha; Valpha.zeros(NSAMP);
  /* Latent variable */
  arma::mat probit_theta_latent; probit_theta_latent.zeros(NSITE, NSP);
  arma::mat probit_theta_pred; probit_theta_pred.zeros(NCELL, NSP);
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
  arma::mat probit_theta_run ;probit_theta_run.zeros(NSITE,NSP);
  arma::mat probit_theta_pred_run ;probit_theta_pred_run.zeros(NCELL,NSP);
  // data 
  arma::mat data_pred = arma::join_rows(X_pred,W_run);
  arma::mat data = mat_select_lines(data_pred,cells);
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
        
        // Mean of the prior
        double probit_theta = arma::as_scalar(data.row(i)*param_run.col(j)  + alpha_run(cells(i)));
        
        // Actualization
        if ( Y(i,j) == 1) {
          Z_run(i,j) = rtnorm(s,0,R_PosInf,probit_theta_run(i,j), 1);
        }
        
        if ( Y(i,j) == 0) {
          Z_run(i,j) = rtnorm(s,R_NegInf,0,probit_theta_run(i,j), 1);
        }
      }
    }
    
    arma::vec alpha_run_sites = vec_select_elems(alpha_run,cells);
    //////////////////////////////////
    // mat param: Gibbs algorithm //
    
    // Loop on species
    for (int j=0; j<NSP; j++) {
      // small_v
      arma::vec small_v = inv(Vparam)*muparam + data.t()*(Z_run.col(j) - alpha_run_sites);
      // big_V
      arma::mat big_V = inv(inv(Vparam)+data.t()*data);
      
      // Draw in the posterior distribution
      arma::vec param_prop = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
      
      // constraints on lambda
      for (int l=0; l<NL; l++) {
        if (l > j) {
          param_prop(NP+l) = 0;
        }
        if ((l==j) & (param_prop(NP+l)< 0)) {
          param_prop(NP+l) = param_run(NP+l,j);
        }
      }
      param_run.col(j) = param_prop;
    }
    
    /////////////////////////////////////////////
    // mat latent variable W: Gibbs algorithm //
    arma::mat beta_run = param_run.submat(0,0,NP-1,NSP-1);
    arma::mat lambda_run = param_run.submat(NP,0,NP+NL-1,NSP-1);
    arma::rowvec Wn_sum; Wn_sum.zeros(NL);
    
    // Loop on cells
    for (int n=0; n<NCELL; n++) {
      int nNeighbors = n_neighbors(n);
      arma::vec sumNeighbors; sumNeighbors.zeros(NL);
      arma::uvec Id_cell = find(mat_neighbors.col(0)==n);
      for (int m=0;m<nNeighbors;m++) {
        sumNeighbors += W_run.row(mat_neighbors(Id_cell(m),1)).t();
      }
      arma::vec meanNeighbors = sumNeighbors * 1/nNeighbors; 
      
      /* W_visited */
      if (viscell[n]>0) {
        //site 
        arma::uvec sites = find(cells == n);
        // big_V
        arma::mat big_V = inv(inv(VW*1/nNeighbors) + lambda_run*lambda_run.t());
        // small_v
        arma::vec small_v =  (VW*1/nNeighbors)*meanNeighbors + lambda_run*(Z_run.row(sites(0))-X.row(sites(0))*beta_run-alpha_run(cells(sites(0)))).t();
        
        // Draw in the posterior distribution
        arma::vec W_n = arma_mvgauss(s, big_V*small_v, chol_decomp(big_V));
        W_run.row(n) = W_n.t();
      }
      
      /* W_unvisited */
      else {
        arma::vec W_n = arma_mvgauss(s, meanNeighbors, chol_decomp(VW*1/nNeighbors));
        W_run.row(n) = W_n.t();
      }
      Wn_sum += W_run.row(n);
    }
    
    /* Centering W_run.row(n) */
    arma::rowvec Wn_bar = Wn_sum/NCELL;
    for (int n=0; n<NCELL; n++) {
      W_run.row(n)=W_run.row(n)-Wn_bar;
    }
    
    data_pred = arma::join_rows(X_pred,W_run);
    data = mat_select_lines(data_pred,cells);
    
    //////////////////////////////////
    // vec alpha : Gibbs algorithm //
    
    double alpha_sum=0.0;
    // Loop on cells
    for (int n=0; n<NCELL; n++) {
      int nNeighbors = n_neighbors(n);
      double sumNeighbors = 0.0;
      arma::uvec Id_cell = find(mat_neighbors.col(0)==n);
      for (int m=0;m<nNeighbors;m++) {
        sumNeighbors +=alpha_run(mat_neighbors(Id_cell(m),1));
      }
      double meanNeighbors = sumNeighbors/nNeighbors; 
      
      /* alpha_visited */
      if (viscell[n]>0) {
        //site 
        arma::uvec sites = find(cells == n);
        // small_v
        double small_v = (Valpha_run/nNeighbors)*meanNeighbors + arma::sum(Z_run.row(sites(0))-data.row(sites(0))*param_run);
        // big_V
        double big_V = 1/(nNeighbors/Valpha_run + NSP);
        // Draw in the posterior distribution
        alpha_run(n) = big_V*small_v + gsl_ran_gaussian_ziggurat(s, std::sqrt(big_V));
      }
      
      /* alpha_unvisited */
      else {
        alpha_run(n) = meanNeighbors + gsl_ran_gaussian_ziggurat(s, std::sqrt(Valpha_run/nNeighbors));
      }
      
      alpha_sum += alpha_run(n);
    }
    
    /* Centering alpha_run(n) */
    double alpha_bar=alpha_sum/NCELL;
    for (int n=0; n<NCELL; n++) {
      alpha_run(n)=alpha_run(n)-alpha_bar;
    }
    double Sum =0.0; 
    for (int n=0; n<NCELL; n++) {
      double Sum_neigh=0.0;
      double nNeigh= n_neighbors(n);
      double alpha_n = alpha_run(n);
      arma::uvec Id_cell = find(mat_neighbors.col(0)==n);
      for (int m=0; m<nNeigh; m++) {
        Sum_neigh += alpha_run(mat_neighbors(Id_cell(m),1));
      }
      Sum += alpha_n*(nNeigh*alpha_n-Sum_neigh);
    }
    
    ////////////////////////////////////////////////
    // Valpha
    double shape_posterior = shape + 0.5*(NCELL-1);
    double rate_posterior = rate + 0.5*Sum;
    
    Valpha_run = rate_posterior/gsl_ran_gamma_mt(s, shape_posterior, 1.0);
    
    
    //////////////////////////////////////////////////
    // Predictions
    for (int n=0; n<NPRED; n++) {
      /* probit_theta_pred_run */
      for (int j=0; j<NSP; j++) {
        probit_theta_pred_run(n,j) = arma::as_scalar(data_pred.row(n)*param_run.col(j)  + alpha_run(n));
      }
    }
    //////////////////////////////////////////////////
    //// Deviance
    
    // logLikelihood
    double logL = 0.0;
    for ( int i = 0; i < NSITE; i++ ) {
      for ( int j = 0; j < NSP; j++ ) {
        // probit(theta_ij) = X_i*beta_j + W_i*lambda_j + alpha_i 
        probit_theta_run(i,j) = arma::as_scalar(data_pred.row(cells(i))*param_run.col(j)  + alpha_run(cells(i)));
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
        param.tube(isamp-1,j) = param_run.col(j);
        for ( int i=0; i<NSITE; i++ ) {
          Z_latent(i,j) += Z_run(i,j) / NSAMP; // We compute the mean of NSAMP values
          probit_theta_latent(i,j) += probit_theta_run(i,j)/NSAMP;  
        }
        for (int n=0; n<NCELL; n++) {
          probit_theta_pred = probit_theta_pred_run(n)/NSAMP;
          W.tube(isamp-1,n) = W_run.row(n);
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
    
    
    //////////////////////////////////////////////////
    // User interrupt
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
                                          Rcpp::Named("probit_theta_latent") = probit_theta_latent,
                                          Rcpp::Named("probit_theta_pred") = probit_theta_pred);  
  return results;
  
} // end Rcpp_jSDM_probit_block

// Test
/*** R
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
# ===================================================
# Data
# ===================================================
library(raster) 
data(altitude,package="hSDM")
# Number of observation sites 
nsite <- 200 
sites <- c(1:nsite)
nsp <- 20
# Set seed for repeatability 
seed <- 1234
alt.orig <- rasterFromXYZ(altitude) 
extent(alt.orig) <- c(0,50,0,50) 
plot(alt.orig) 
# Center and scale altitudinal data
alt <- scale(alt.orig,center=TRUE,scale=TRUE) 
coords <- coordinates(alt)
plot(alt)
# Rasters must be projected to correctly compute the neighborhood
crs(alt) <- '+proj=utm +zone=1'
ncells <- ncell(alt)
# Matrix of covariates (including the intercept) 
X <- cbind(rep(1,ncells),values(alt))
np <- ncol(X)

# Neighborhood matrix 
neighbors.mat <- adjacent(alt, cells=c(1:ncells), directions=8, pairs=TRUE, sorted=TRUE) 
# Number of neighbors by cell 
n.neighbors <- as.data.frame(table(as.factor(neighbors.mat[,1])))[,2] 
# Adjacent cells
adj <- neighbors.mat[,2] 
# Generate symmetric adjacency matrix, A 
A <- matrix(0,ncells,ncells) 
index.start <- 1 
for (i in 1:ncells) { 
  index.end <- index.start+n.neighbors[i]-1 
  A[i,adj[c(index.start:index.end)]] <- 1 
  index.start <- index.end+1 
}

# Sample the observations in the landscape 
set.seed(seed) 
x.coord <- runif(nsite,0,50) 
set.seed(2*seed) 
y.coord <- runif(nsite,0,50) 
library(sp) 
sites.sp <- SpatialPoints(coords=cbind(x.coord,y.coord)) 
cells <- extract(alt,sites.sp,cell=TRUE)[,1]
# Extract altitude data for sites 
alt.sites <- extract(alt,sites.sp) 
# Generate spatial random effects
Valpha.target <- 0.5 # Variance of spatial random effects 
d <- 1 # Spatial dependence parameter = 1 for intrinsic CAR 
Q <- diag(n.neighbors)-d*A + diag(.0001,ncells) 
# Add small constant to make Q non-singular 
covalpha <- Valpha.target*solve(Q) # Covariance of alphas 
alpha <- c(rmvn(1,mu=rep(0,ncells),V=covalpha,seed=seed)) # Spatial Random Effects 
alpha.target <- alpha-mean(alpha) # Centering alphas on zeros
alpha.rast <- rasterFromXYZ(xyz=cbind(coords,alpha))
alpha.sites <- extract(alpha.rast,sites.sp)
# Matrix of Latent variables W
covW <- 1*solve(Q) # Covariance of latent variables W_1 and W_2 
W <- cbind(rmvn(1,mu=rep(0,ncells),V=covW,seed=seed),rmvn(1,mu=rep(0,ncells),V=covW,seed=seed))
nl <- ncol(W)
# Generate species effects betas and lambas on bioclimatic and latent variables
beta.target <- t(matrix(runif(nsp*np,-2,2), byrow=TRUE, nrow=nsp))
l.zero <- 0
l.diag <- runif(2,0,1)
l.other <- runif(nsp*2-3,-1,1)
lambda.target <- t(matrix(c(l.diag[1],l.zero,l.other[1],l.diag[2],l.other[-1]), byrow=T, nrow=nsp))
param.target <- rbind(beta.target,lambda.target)
# Compute the probability of presence theta for these observations 
probit.theta.cells <- X %*% beta.target +  W %*% lambda.target + alpha.target
probit.theta.rast <- rasterFromXYZ(cbind(coords,probit.theta.cells))
# probit.theta.sites <- extract(probit.theta.rast,sites.sp)
X.sites <- cbind(rep(1,nsite),alt.sites) 
W.rast <- rasterFromXYZ(cbind(coords,W))
W.sites <- extract(W.rast,sites.sp)
e = matrix(rnorm(nsp*ncells,0,1),ncells,nsp)
probit.theta.site <- X.sites %*% beta.target + W.sites %*% lambda.target + alpha.sites
Z.true.sites <- probit.theta.sites + e[cells,]
# Generate latent variables 
Z.true.cells <- probit.theta.cells + e

# Simulate observations 
Y.cells = matrix (NA, ncells,nsp)
for (i in 1:ncells){
  for (j in 1:nsp){
    if ( Z.true.cells[i,j] > 0) {Y.cells[i,j] <- 1}
    else {Y.cells[i,j] <- 0}
  }
}
Y.rast <- rasterFromXYZ(cbind(coords,Y.cells))
Y.sites <- extract(Y.rast,sites.sp)
# Group explicative and response variables in a data-frame 
data.obs.df <- data.frame(Y=Y.sites,alt=X.sites[,2]) 
# Transform observations in a spatial object 
data.obs <- SpatialPointsDataFrame(coords=coordinates(sites.sp), data=data.obs.df) 
data.pred <- data.frame(alt=values(alt))

# Plot observations 
#palette() #to see which colors are available 
#mycols <- colors() #to get more colors
#cols <- mycols[sample(c(1:657),20)] #to get different colors
#palette(cols)
# palette("default") reset to default
for (j in 1:nsp){
  par(mfrow=c(1,1))
  plot(alt.orig, main=paste0("species_",j)) 
  points(data.obs[data.obs[[paste0("Y.layer.",j)]]==1,],pch=16)
  points(data.obs[data.obs[[paste0("Y.layer.",j)]]==0,],pch=1)
  legend("topleft",legend=c("presence","absence") ,pch=c(16,1),cex=0.5)
}
param_start=matrix(0,np+nl,nsp)
for (i in 1:nl){
  param_start[np+i,i] = 1
}

# Call to C++ function
# Iterations
nsamp <- 5000
nburn <- 20000
nthin <- 5
ngibbs <- nsamp+nburn
mod <- Rcpp_jSDM_probit_block(ngibbs=ngibbs, nthin=nthin, nburn=nburn,
                              Y=Y.sites, X=X.sites, cells=cells-1,
                              X_pred=X, cells_pred=c(1:ncells)-1,
                              n_neighbors=n.neighbors-1, mat_neighbors=neighbors.mat-1, 
                              param_start=param_start, Vparam=diag(c(rep(1.0E6,np),rep(10,nl))),
                              muparam = rep(0,np+nl), W_start=matrix(0,ncells,nl), VW=diag(rep(1,nl)),
                              alpha_start=rep(0,ncells),Valpha_start=1, shape = 0.5, rate = 0.0005, seed=123, verbose=1)

# ===================================================
# Result analysis
# ===================================================

# Parameter estimates
## probit_theta
par(mfrow=c(1,1))
plot(probit.theta.sites,mod$probit_theta_latent)
abline(a=0,b=1,col='red')
## Z
plot(Z.true.sites,mod$Z_latent)
abline(a=0,b=1,col='red')
## alpha
MCMC_alpha <- coda::mcmc(mod$alpha, start=nburn+1, end=ngibbs, thin=nthin)
plot(alpha.target,summary(MCMC_alpha)[[1]][,"Mean"], ylab ="alpha.estimated")
abline(a=0,b=1,col='red')
plot(alpha.sites,summary(MCMC_alpha)[[1]][cells,"Mean"], ylab ="alpha.estimated")
abline(a=0,b=1,col='red')
## Valpha
MCMC_Valpha <- coda::mcmc(mod$Valpha, start=nburn+1, end=ngibbs, thin=nthin)
summary(MCMC_Valpha)
par(mfrow=c(1,2))
coda::traceplot(MCMC_Valpha)
coda::densplot(MCMC_Valpha)
abline(v=Valpha.target,col='red')

## beta_j
par(mfrow=c(np,2))
for (j in 1:4) {
  for (p in 1:np) {
    MCMC.betaj <- coda::mcmc(mod$param[,j,1:np], start=nburn+1, end=ngibbs, thin=nthin)
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
    MCMC.lambdaj <- coda::mcmc(mod$param[,j,(np+1):(nl+np)], start=nburn+1, end=ngibbs, thin=nthin)
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
