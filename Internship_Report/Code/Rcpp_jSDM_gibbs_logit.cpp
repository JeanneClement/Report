#include <RcppArmadillo.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

/* Function logit */
double logit (double x) {
  return std::log(x) - std::log(1-x);
}

/* Function invlogit */
double invlogit (double x) {
  if (x > 0) {
    return 1 / (1 + std::exp(-x));
  }
  else {
    return std::exp(x) / (1 + std::exp(x));
  }
}

/* Function mean */
double mean (arma::vec x){
  int n = x.size();
  double sum = 0.0;
  for ( int i = 0; i < n ; i++ ) {
    sum += x(i);
  }
  return sum/n ;
}

/* Function var */
double var (arma::vec x){
  int n = x.size();
  double sum = 0.0;
  for ( int i = 0; i < n ; i++ ) {
    sum += (x(i) - mean(x))*(x(i) - mean(x));
  }
  return sum/n ;
}

/* dens_par */
struct dens_par {
  // Data 
  int NSITE;
  int NSP;
  arma::umat Y;
  arma::umat T;
  // Suitability 
  // beta
  int NP;
  arma::mat X;
  int pos_beta;
  int sp_beta;
  arma::mat mubeta;
  arma::mat Vbeta;
  arma::mat beta_run;
  // lambda
  int NL; 
  int pos_lambda;
  int sp_lambda;
  arma::mat mulambda;
  arma::mat Vlambda;
  arma::mat lambda_run;
  // W
  int site_W;
  int pos_W;
  arma::mat muW;
  arma::mat VW;
  arma::mat W_run;
  //alpha
  int site_alpha; 
  double Valpha_run;
  double shape;
  double rate;
  arma::rowvec alpha_run;
};

/* betadens */
double betadens (double beta_jk, void *dens_data) {
  // Pointer to the structure: d
  dens_par *d;
  d = static_cast<dens_par *> (dens_data);
  // Indicating the rank and the species of the parameter of interest
  int k = d->pos_beta;
  int j = d->sp_beta;
  // logLikelihood
  double logL = 0.0;
  for ( int i = 0; i < d->NSITE; i++ ) {
    /* theta */
    double Xpart_theta = 0.0;
    for ( int p = 0; p < d->NP; p++ ) {
      if ( p != k ) {
        Xpart_theta += d->X(i,p) * d->beta_run(j,p);
      }
    }
    for ( int q = 0; q < d->NL; q++ ) {
      Xpart_theta += d->W_run(i,q) * d->lambda_run(j,q);
    } 
    Xpart_theta += d->X(i,k) * beta_jk + d->alpha_run(i);
    double theta = invlogit(Xpart_theta);
    /* log Likelihood */
    logL += R::dbinom(d->Y(i,j), d->T(i,j), theta, 1);
  } // loop on sites 
  
  // logPosterior = logL + logPrior
  double logP = logL + R::dnorm(beta_jk, d->mubeta(j,k), std::sqrt(d->Vbeta(j,k)), 1);
  return logP;
}

/* lambdadens */
double lambdadens (double lambda_jq, void *dens_data) {
  // Pointer to the structure: d
  dens_par *d;
  d = static_cast<dens_par *> (dens_data);
  // Indicating the rank and the species of the parameter of interest
  int q = d->pos_lambda;
  int j = d->sp_lambda;
  // logLikelihood
  double logL = 0.0;
  for ( int i = 0; i < d->NSITE; i++ ) {
    /* theta */
    double Xpart_theta = 0.0;
    for ( int p = 0;  p < d->NP; p++ ) {
      Xpart_theta += d->X(i,p) * d->beta_run(j,p);
    }
    for ( int l = 0;  l < d->NL; l++ ) {
      if(l != q) {
        Xpart_theta += d->W_run(i,l)* d->lambda_run(j,l);
      }
    }
    Xpart_theta += d->W_run(i,q) * lambda_jq +  d->alpha_run(i);
    double theta = invlogit(Xpart_theta);
    /* log Likelihood */
    logL += R::dbinom(d->Y(i,j), d->T(i,j), theta, 1);
  } // loop on sites 
  
  // logPosterior = logL + logPrior
  double logP = logL + R::dnorm(lambda_jq, d->mulambda(j,q), std::sqrt(d->Vlambda(j,q)), 1);
  return logP;
}

double lambdaUdens (double lambda_jq, void *dens_data) {
  // Pointer to the structure: d
  dens_par *d;
  d = static_cast<dens_par *> (dens_data);
  // Indicating the rank and the species of the parameter of interest
  int q = d->pos_lambda;
  int j = d->sp_lambda;
  // logLikelihood
  double logL = 0.0;
  for ( int i = 0; i < d->NSITE; i++ ) {
    /* theta */
    double Xpart_theta = 0.0;
    for ( int p = 0;  p < d->NP; p++ ) {
      Xpart_theta += d->X(i,p) * d->beta_run(j,p);
    }
    for ( int l = 0;  l < d->NL; l++ ) {
      if(l != q) {
        Xpart_theta += d->W_run(i,l)* d->lambda_run(j,l);
      }
    }
    Xpart_theta += d->W_run(i,q) * lambda_jq + d->alpha_run(i);
    double theta = invlogit(Xpart_theta);
    /* log Likelihood */
    logL += R::dbinom(d->Y(i,j), d->T(i,j), theta, 1);
  } // loop on sites 
  
  // logPosterior = logL + logPrior
  double logP = logL + R::dunif(lambda_jq, d->mulambda(j,q), d->Vlambda(j,q), 1);
  return logP;
}

/* Wdens */
double Wdens (double W_iq, void *dens_data) {
  // Pointer to the structure: d
  dens_par *d;
  d = static_cast<dens_par *> (dens_data);
  // Indicating the rank and the species of the parameter of interest
  int i = d->site_W;
  int q = d->pos_W;
  // logLikelihood
  double logL = 0.0;
  for ( int j = 0; j < d->NSP; j++ ) {
    /* theta */
    double Xpart_theta = 0.0;
    for ( int p = 0;  p < d->NP; p++ ) {
      Xpart_theta += d->X(i,p) * d->beta_run(j,p);
    }
    for ( int l = 0;  l < d->NL; l++ ) {
      if(l != q) {
        Xpart_theta += d->W_run(i,l)* d->lambda_run(j,l);
      }
    }
    Xpart_theta += W_iq * d->lambda_run(j,q) +  d->alpha_run(i);
    double theta = invlogit(Xpart_theta);
    /* log Likelihood */
    logL += R::dbinom(d->Y(i,j), d->T(i,j), theta, 1);
  } // loop on species
  
  // logPosterior = logL + logPrior
  double logP = logL + R::dnorm(W_iq, d->muW(i,q), std::sqrt(d->VW(i,q)), 1);
  return logP;
}

/* alphadens */

double alphadens (double alpha_i, void *dens_data) {
  // Pointer to the structure: d
  dens_par *d;
  d = static_cast<dens_par *> (dens_data);
  // Indicating the site of the parameter of interest
  int i = d->site_alpha;
  // logLikelihood
  double logL = 0.0;
  /* theta */
  for ( int j = 0; j < d->NSP; j++ ) {
    double Xpart_theta = 0.0;
    for ( int p = 0; p < d->NP; p++ ) {
      Xpart_theta += d->X(i,p) * d->beta_run(j,p); 
    } 
    for ( int q = 0; q < d->NL; q++ ) {
      Xpart_theta += d->W_run(i,q) * d->lambda_run(j,q);
    } 
    Xpart_theta += alpha_i; 
    double theta = invlogit(Xpart_theta);
    /* log Likelihood */
    logL += R::dbinom(d->Y(i,j), d->T(i,j), theta, 1);
  } // loop on species
  
  // logPosterior = logL + logPrior
  double logP = logL + R::dnorm(alpha_i, 0, std::sqrt(d->Valpha_run),1);
  return logP;
}

/* ************************ */
/* Gibbs sampler function */

// [[Rcpp::export]]
Rcpp::List  Rcpp_jSDM_gibbs_logit(
    const int ngibbs, int nthin, int nburn, // Number of iterations, burning and samples
    arma::umat Y, // Number of successes (presences)
    arma::umat T, // Number of trials
    arma::mat X, // Suitability covariates
    arma::mat beta_start,//beta
    arma::mat mubeta,
    arma::mat Vbeta,
    arma::mat mulambda,//lambda
    arma::mat Vlambda,
    arma::mat lambda_start,
    arma::mat muW,//W
    arma::mat VW,
    arma::mat W_start,
    arma::vec alpha_start,//alpha
    double Valpha_start,
    double shape,
    double rate,
    const int seed,
    const double ropt,
    const int verbose) {
  
  ////////////////////////////////////////
  // Defining and initializing objects //
  
  ////////////////////////////////////////
  // Initialize random number generator //
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, seed);
  
  ///////////////////////////
  // Redefining constants //
  const int NGIBBS = ngibbs;
  const int NTHIN = nthin;
  const int NBURN = nburn;
  const int NSAMP = (NGIBBS-NBURN)/NTHIN;
  const int NSITE = X.n_rows;
  const int NP = X.n_cols;
  const int NSP = Y.n_cols;
  const int NL = lambda_start.n_cols;
  
  ////////////////////////////////////////////
  // Declaring new objects to store results //
  /* Parameters */
  arma::Cube<double> beta; beta.zeros(NSAMP, NSP, NP);
  arma::Cube<double> lambda; lambda.zeros(NSAMP, NSP, NL);
  arma::Cube<double> W; W.zeros(NSAMP, NSITE, NL);
  arma::mat alpha; alpha.zeros(NSAMP, NSITE);
  arma::vec Valpha; Valpha.zeros(NSAMP);
  /* Latent variable */
  arma::mat theta_run; theta_run.zeros(NSITE, NSP);
  arma::mat theta_latent; theta_latent.zeros(NSITE, NSP);
  /* Deviance */
  arma::vec Deviance; Deviance.zeros(NSAMP);
  
  //////////////////////////////////////////////////////////
  // Set up and initialize structure for density function //
  dens_par dens_data;
  // Data 
  dens_data.NSITE = NSITE;
  dens_data.NSP = NSP;
  dens_data.NL = NL;
  // Y
  dens_data.Y = Y;
  // T
  dens_data.T = T;
  // Suitability process 
  dens_data.NP = NP;
  dens_data.X = X;
  // beta
  dens_data.pos_beta = 0;
  dens_data.sp_beta = 0;
  dens_data.mubeta = mubeta;
  dens_data.Vbeta = Vbeta;
  dens_data.beta_run = beta_start;
  // lambda 
  dens_data.pos_lambda = 0;
  dens_data.sp_lambda = 0;
  dens_data.mulambda = mulambda;
  dens_data.Vlambda = Vlambda;
  dens_data.lambda_run = lambda_start;
  // W
  dens_data.site_W = 0;
  dens_data.pos_W = 0;
  dens_data.muW = muW;
  dens_data.VW = VW;
  dens_data.W_run = W_start;
  // alpha
  dens_data.site_alpha = 0;
  dens_data.shape = shape;
  dens_data.rate = rate;
  dens_data.Valpha_run = Valpha_start;
  dens_data.alpha_run = alpha_start.t();
  
  ////////////////////////////////////////////////////////////
  // Proposal variance and acceptance for adaptive sampling //
  
  // beta
  arma::mat sigmap_beta; sigmap_beta.ones(NSP,NP);
  arma::mat nA_beta; nA_beta.zeros(NSP,NP);
  arma::mat Ar_beta; Ar_beta.zeros(NSP,NP); // Acceptance rate
  
  // lambda
  arma::mat sigmaq_lambda; sigmaq_lambda.ones(NSP,NL);
  arma::mat nA_lambda; nA_lambda.zeros(NSP,NL);
  arma::mat Ar_lambda; Ar_lambda.zeros(NSP,NL); // Acceptance rate
  
  // W
  arma::mat sigmaq_W; sigmaq_W.ones(NSITE,NL);
  arma::mat nA_W; nA_W.zeros(NSITE,NL);
  arma::mat Ar_W; Ar_W.zeros(NSITE,NL); // Acceptance rate
  
  // alpha
  arma::vec sigma_alpha; sigma_alpha.ones(NSITE);
  arma::vec nA_alpha; nA_alpha.zeros(NSITE);
  arma::vec Ar_alpha; Ar_alpha.zeros(NSITE); // Acceptance rate
  
  
  ////////////
  // Message//
  Rprintf("\nRunning the Gibbs sampler. It may be long, please keep cool :)\n\n");
  R_FlushConsole();
  
  ////////////////////
  // Gibbs sampler //
  
  for ( int g = 0; g < NGIBBS; g++ ) {
    
    double sum = 0.0;
    for ( int i = 0; i < NSITE; i++ ) {
      // alpha
      dens_data.site_alpha = i; // Specifying the site 
      double x_now = dens_data.alpha_run(i);
      double x_prop = x_now + gsl_ran_gaussian_ziggurat(r, sigma_alpha(i));
      double p_now = alphadens(x_now, &dens_data);
      double p_prop = alphadens(x_prop, &dens_data);
      double ratio = std::exp(p_prop - p_now); // ratio
      double z = gsl_rng_uniform(r);
      // Actualization
      if ( z < ratio ) {
        dens_data.alpha_run(i) = x_prop;
        nA_alpha(i)++;
      } 
      sum += dens_data.alpha_run(i)*dens_data.alpha_run(i);
      
      // W
      dens_data.site_W = i; // Specifying the site
      for ( int q = 0; q < NL; q++ ) {
        dens_data.pos_W = q; // Specifying the rank of the latent variable of interest
        double x_now = dens_data.W_run(i,q);
        double x_prop = x_now + gsl_ran_gaussian_ziggurat(r,sigmaq_W(i,q));
        double p_now = Wdens(x_now, &dens_data);
        double p_prop = Wdens(x_prop, &dens_data);
        double ratio = std::exp(p_prop - p_now); // ratio
        double z = gsl_rng_uniform(r);
        // Actualization
        if ( z < ratio ) {
          dens_data.W_run(i,q) = x_prop;
          nA_W(i,q)++;
        } 
      } // loop on rank of latent variable 
    } // loop on sites 
    
    // Valpha
    double shape_posterior = dens_data.shape + 0.5*NSITE;
    double rate_posterior = dens_data.rate + 0.5*sum;
    
    dens_data.Valpha_run = rate_posterior/gsl_ran_gamma_mt(r, shape_posterior, 1.0);
    
    // Centering and reducing W_i 
    for ( int i = 0; i < NSITE; i++ ) {
      for ( int q = 0; q < NL; q++ ) {
        dens_data.W_run(i,q) = dens_data.W_run(i,q) - mean(dens_data.W_run.col(q));
        dens_data.W_run(i,q) = dens_data.W_run(i,q)/std::sqrt(var(dens_data.W_run.col(q)));
      }
    }
    
    for ( int j = 0; j < NSP; j++ ) {
      // beta
      dens_data.sp_beta = j; // Specifying the species
      for ( int p = 0; p < NP; p++ ) {
        dens_data.pos_beta = p; // Specifying the rank of the parameter of interest
        double x_now = dens_data.beta_run(j,p);
        double x_prop = x_now + gsl_ran_gaussian_ziggurat(r, sigmap_beta(j,p));
        double p_now = betadens(x_now, &dens_data);
        double p_prop = betadens(x_prop, &dens_data);
        double ratio = std::exp(p_prop - p_now); // ratio
        double z = gsl_rng_uniform(r);
        // Actualization
        if ( z < ratio ) {
          dens_data.beta_run(j,p) = x_prop;
          nA_beta(j,p)++;
        } 
      } // loop on rank of parameters
      
      // lambda 
      dens_data.sp_lambda = j; // Specifying the species
      for ( int q = 0; q < NL; q++ ) {
        dens_data.pos_lambda = q ; // Specifying the rank of the parameter of interest
        if (q < j ) {
          double x_now = dens_data.lambda_run(j,q);
          double x_prop = x_now + gsl_ran_gaussian_ziggurat(r, sigmaq_lambda(j,q));
          double p_now = lambdadens(x_now, &dens_data);
          double p_prop = lambdadens(x_prop, &dens_data);
          double ratio = std::exp(p_prop - p_now); // ratio
          double z = gsl_rng_uniform(r);
          // Actualization
          if ( z < ratio ) {
            dens_data.lambda_run(j,q) = x_prop;
            nA_lambda(j,q)++;
          }
        }
        if (q == j) { 
          double x_now = dens_data.lambda_run(j,q);
          double x_prop = x_now + gsl_ran_gaussian_ziggurat(r,sigmaq_lambda(j,q));
          double p_now = lambdaUdens(x_now, &dens_data);
          double p_prop = lambdaUdens(x_prop, &dens_data);
          double ratio = std::exp(p_prop - p_now); // ratio
          double z = gsl_rng_uniform(r);
          // Actualization
          if ( z < ratio ) {
            dens_data.lambda_run(j,q) = x_prop;
            nA_lambda(j,q)++;
          }  
        }
        if (q > j) { 
          dens_data.lambda_run(j,q) = 0;
        } 
      } // loop on rank of latent variable
    } // loop on species
    
    ///////////////
    // Deviance //
    
    // logLikelihood
    double logL = 0.0;
    for ( int i = 0; i < NSITE; i++ ) {
      for ( int j = 0; j < NSP; j++ ) {
        /* theta */
        double Xpart_theta = 0.0;
        for ( int p = 0; p < NP; p++ ) {
          Xpart_theta += dens_data.X(i,p) * dens_data.beta_run(j,p);
        }
        for ( int q = 0; q < NL; q++ ) {
          Xpart_theta += dens_data.W_run(i,q) * dens_data.lambda_run(j,q);
        }
        Xpart_theta += dens_data.alpha_run(i);
        theta_run(i,j) = invlogit(Xpart_theta);
        /* log Likelihood */
        logL += R::dbinom(dens_data.Y(i,j), dens_data.T(i,j), theta_run(i,j), 1);
      } // loop on species
    } // loop on sites
    
    // Deviance
    double Deviance_run = -2 * logL;
    
    
    /////////////
    // Output //
    if (((g+1)>NBURN) && (((g+1)%(NTHIN))==0)) {
      int isamp=((g+1)-NBURN)/(NTHIN);
      for ( int j=0; j<NSP; j++ ) {
        beta.tube(isamp-1,j) = dens_data.beta_run.row(j);
        lambda.tube(isamp-1,j) = dens_data.lambda_run.row(j);
        for ( int i=0; i<NSITE; i++ ) {
          W.tube(isamp-1,i) = dens_data.W_run.row(i);
          theta_latent(i,j) += theta_run(i,j) / NSAMP; // We compute the mean of NSAMP values
        }//loop on sites
      }// loop on species
      alpha.row(isamp-1) = dens_data.alpha_run;
      Valpha(isamp-1) = dens_data.Valpha_run;
      Deviance(isamp-1) = Deviance_run;
    }
    
    ///////////////////////////////////////////////
    // Adaptive sampling (on the burnin period) //
    const double ROPT = ropt;
    int DIV = 0;
    if ( NGIBBS >= 1000 ) DIV=100;
    else DIV = NGIBBS / 10;
    /* During the burnin period */
    if ( (g+1)%DIV== 0 && (g+1)<=NBURN ) {
      for (int j=0; j<NSP; j++) {
        for ( int p=0; p<NP; p++ ) {
          Ar_beta(j,p) = ((double) nA_beta(j,p)) / DIV;
          if ( Ar_beta(j,p) >= ROPT )
            sigmap_beta(j,p) = sigmap_beta(j,p)*(2-(1-Ar_beta(j,p)) / (1-ROPT));
          else sigmap_beta(j,p) = sigmap_beta(j,p) / (2-Ar_beta(j,p) / ROPT);
          nA_beta(j,p) = 0.0; // We reinitialize the number of acceptance to zero for beta
        } // loop on rank of parameters
        for ( int q=0; q<NL; q++ ) {
          Ar_lambda(j,q) = ((double) nA_lambda(j,q)) / DIV;
          if ( Ar_lambda(j,q) >= ROPT ) 
            sigmaq_lambda(j,q) = sigmaq_lambda(j,q)*(2-(1-Ar_lambda(j,q)) / (1-ROPT));
          else sigmaq_lambda(j,q) = sigmaq_lambda(j,q) / (2-Ar_lambda(j,q) / ROPT);
          nA_lambda(j,q) = 0.0; // We reinitialize the number of acceptance to zero for lambda 
        } // loop on rank of latent variable
      } // loop on species 
      for (int i=0; i<NSITE; i++) {
        Ar_alpha(i) = ((double) nA_alpha(i)) / DIV;
        if ( Ar_alpha(i) >= ROPT ) sigma_alpha(i) = sigma_alpha(i) * (2-(1-Ar_alpha(i)) / (1-ROPT));
        else sigma_alpha(i) = sigma_alpha(i) / (2-Ar_alpha(i) / ROPT);
        nA_alpha(i) = 0.0; // We reinitialize the number of acceptance for alpha to zero
        for ( int q=0; q<NL; q++ ) {
          Ar_W(i,q) = ((double) nA_W(i,q)) / DIV;
          if ( Ar_W(i,q) >= ROPT ) sigmaq_W(i,q) = sigmaq_W(i,q) * (2-(1-Ar_W(i,q)) / (1-ROPT));
          else sigmaq_W(i,q) = sigmaq_W(i,q) / (2-Ar_W(i,q) / ROPT);
          nA_W(i,q) = 0.0; // We reinitialize the number of acceptance to zero for z
        } // loop on rank of latent variable
      } // loop on sites
    }
    
    /* After the burnin period */
    if ( (g+1) % DIV == 0 && (g+1) > NBURN ) {
      for (int j=0; j<NSP; j++) {
        for (int p=0; p<NP; p++) {
          Ar_beta(j,p) = ((double) nA_beta(j,p)) / DIV;
          nA_beta(j,p) = 0.0; // We reinitialize the number of acceptance to zero for beta
        } // loop on rank of parameters
        for (int q=0; q<NL; q++) {
          Ar_lambda(j,q) = ((double) nA_lambda(j,q)) / DIV;
          nA_lambda(j,q) = 0.0; // We reinitialize the number of acceptance to zero for lambda
        } // loop on rank of latent variable 
      } // loop on species
      for (int i=0; i<NSITE; i++) {
        Ar_alpha(i) = ((double) nA_alpha(i)) / DIV;
        nA_alpha(i) = 0.0; // We reinitialize the number of acceptance for alpha to zero
        for (int q=0; q<NL; q++) {
          Ar_W(i,q) = ((double) nA_W(i,q)) / DIV;
          nA_W(i,q) = 0.0; // We reinitialize the number of acceptance to zero for z
        } // loop on rank of latent variable
      } // loop on sites
    }
    
    //////////////////////////////////////////////////
    // Progress bar
    double Perc = 100 * (g+1) / (NGIBBS);
    if ( (g+1) % (NGIBBS/100) == 0 && verbose == 1) {
      Rprintf("*");
      R_FlushConsole();
      if( (g+1) % (NGIBBS/10) == 0 ) {
        double mAr_beta=0; // Mean acceptance rate of beta
        double mAr_lambda=0; // Mean acceptance rate of lambda
        for ( int j = 0; j < NSP; j++ ) {
          for ( int p = 0; p < NP; p++ ) {
            mAr_beta += Ar_beta(j,p) / (NSP*NP);
          } // loop on rank of parameters
          for ( int q = 0; q < NL; q++ ) {
            mAr_lambda += Ar_lambda(j,q) / (NSP*NL-NL*(NL-1)*0.5);
          } // loop on rank of latent variable 
        } // loop on species
        
        double mAr_W=0; // Mean acceptance rate of W
        double mAr_alpha=0; // Mean acceptance rate of alpha 
        for ( int i = 0; i < NSITE; i++ ) {
          mAr_alpha += Ar_alpha(i) / NSITE;
          for ( int q = 0; q < NL; q++ ) {
            mAr_W += Ar_W(i,q) / (NSITE*NL);
          }// loop on rank of latent variable 
        }// loop on sites
        Rprintf(":%.1f%%, mean accept. rates= beta:%.3f lambda:%.3f W:%.3f alpha:%4.3f\n",
                Perc, mAr_beta, mAr_lambda, mAr_W, mAr_alpha);
        R_FlushConsole();
      }
    }
    
    //////////////////////////////////////////////////
    // User interrupt
    R_CheckUserInterrupt(); // allow user interrupt
    
  } // Gibbs sampler
  
  // Free memory
  gsl_rng_free(r);
  
  // Return results as a Rcpp::List
  Rcpp::List w = Rcpp::List::create(Rcpp::Named("beta") = beta,
                                    Rcpp::Named("lambda") = lambda,
                                    Rcpp::Named("W") = W,
                                    Rcpp::Named("alpha") = alpha,
                                    Rcpp::Named("Valpha") = Valpha,
                                    Rcpp::Named("Deviance") = Deviance,
                                    Rcpp::Named("theta_latent") = theta_latent);
  
  return w;
  
}// end Rcpp_jSDM_gibbs_logit function
