#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;
using namespace std;


inline int sign(const double x) {
  return (x > 0.0) - (x < 0.0);
}

inline double inv_sqrt(const double &x) {
  double y = x;
  double xhalf = (double) 0.5 * y;
  long long i = *(long long*) (&y);
  i = 0x5fe6ec85e7de30daLL - (i >> 1); // LL suffix for (long long) type for GCC
  y = *(double*) (&i);
  y = y *((double) 1.5 - xhalf * y * y);
  return y;
}

//' @title Sparse logistic with lasso penalizer
//' @description Fits a sparse logistic model with lasso penalizer
//' @export
// [[Rcpp::export]]
List sparse_logit(
    const arma::vec &beta0,
    const arma::sp_mat &Xt,
    const arma::vec &y,
    const double lambda = 0.0,
    const int maxepochs = 1,
    const double step_scale = 2.0,
    const double tol = 1e-3,
    const double discount = 0.95) {
  
  // Initialize main values for SGD
  size_t n = Xt.n_cols, p = Xt.n_rows;
  vec beta(beta0), adagrad_gsquared(p), yhat(n);
  adagrad_gsquared.fill(1e-3);
  uvec last_update(p);
  last_update.fill(0);
  bool converged = false;
  double adagrad_g0squared = 0;
  
  // Initialize intercept
  double what = (sum(y) + 1.0) / (n + 2.0);
  double alpha = log(what / (1 - what));
  
  // Initialize aux variables for the iteration
  double regi, errori, gradij, penalizer_debt, weight, betajnew;
  int global_counter, missing_updates;
  arma::uword j;
  
  // Global counter
  global_counter= 0;
  
  // Main loop
  for (int epoch_iter = 0; !converged && epoch_iter < maxepochs; epoch_iter++) {
    for(arma::uword i = 0; i < n; i++) {
      
      // Column limits for the row of X (column of Xt)
      auto xi_start = Xt.begin_col(i), xi_end = Xt.end_col(i);
      
      // Compute fitted value and error efficiently for sparse data
      regi = alpha;
      for (auto xi_it = xi_start; xi_it != xi_end; ++xi_it) {
        regi += (*xi_it) * beta[xi_it.row()];
      }
      yhat[i] = 1 / (1 + exp(-regi));
      errori = y[i] - yhat[i];
      
      // Update intercept and its adagrad gradient sum of squares
      adagrad_g0squared += errori * errori;
      alpha += step_scale * inv_sqrt(adagrad_g0squared) * errori;
        
      // Compute gradients and lazy update beta
      for (auto xi_it = xi_start; xi_it != xi_end; ++xi_it) {
        
        // Feature number (row since we are using the transpose)
        j = xi_it.row();
        
        // Lazy updating debt of penalty
        missing_updates = global_counter - last_update[j] - 1;
        weight = inv_sqrt(adagrad_gsquared[j]);
        penalizer_debt = sign(beta[j]) * missing_updates * step_scale * weight * lambda;
        last_update[j] = global_counter;
        
        // Update AdaGrad weights
        gradij = - errori * (*xi_it);
        adagrad_gsquared[j] += gradij * gradij;
        
        // Update parameters
        weight = inv_sqrt(adagrad_gsquared[j]);
        betajnew = beta[j] - step_scale * weight * gradij;
        beta[j] =  sign(betajnew) * max(0.0, abs(betajnew) - step_scale * weight * lambda - penalizer_debt);
      }
      global_counter++;
    }    
  }
  
  // Output
  return List::create(Named("alpha") = alpha,
                      Named("coefficients") = beta,
                      Named("fitted") = yhat);
}