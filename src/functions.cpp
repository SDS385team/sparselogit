#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

//' @title Sigmoid function with truncation
//' @param u: numeric vector of input
//' @param eps: real number of truncation paramaeter if output is close to 0/1 
// ' @return sigmoid evaluate at input u with truncation
//' @export
// [[Rcpp::export]]
arma::mat sigmoid(const arma::sp_mat & A) {
  // Create a "dense" matrix space for result
  mat out(A.n_rows, A.n_cols);
  // Fill with 0.5 which corresponds to zeros
  out.fill(0.5);
  // Compute prediction for non-zero entries
  for (sp_mat::const_iterator it = A.begin(); it != A.end(); ++it) {
    out(it.row(), it.col()) = 1 / (1 + exp(-*it));
  }
  // Output
  return out;
}


//' @title Negative LogLikelihood of Binomial GLM
//' @param beta: numeric vector of parameters of linear model
//' @param X: numeric matrix with predictor variables
//' @param y: numeric vector with response variable
//' @param m: integer controlling the max number of successes in binomial response y
//' @return the negative loglikelihood evaluated at the input
//' @export
// [[Rcpp::export]]
double nlogl_binom(const arma::sp_mat & beta,
                   const arma::sp_mat & X,
                   const arma::sp_mat & y,
                   const int m = 1,
                   const double lambda = 0.0) {
  int N = X.n_rows;
  mat fitted = sigmoid(X * beta);
  double out = 0.0;
  double eps = 1e-6; // fudge factor for numerical stability
  // loglikelihood
  for (int it = 0; it < N; it++) {
    out -= y[it] * log(max(fitted[it], eps));
    out -= (m - y[it]) * log(min(1 - fitted[it], 1 - eps));
  }
  // penalizer
  if (lambda > 0.0) {
    out += lambda * norm(beta, 1);
  }
  // output
  return out;
}

//' @title Gradient Negative LogLikelihood of Binomial GLM
//' @param beta: numeric vector of parameters of linear model
//' @param X: numeric matrix with predictor variables
//' @param y: numeric vector with response variable
//' @param m: integer controlling the max number of successes in binomial response y
//' @return matrix with one column with the gradient of the negative loglikelihood evaluated at the input
//' @export
// [[Rcpp::export]]
arma::mat nlogl_binom_grad(const arma::sp_mat & beta,
                           const arma::sp_mat & X,
                           const arma::sp_mat & y,
                           const int m = 1,
                           const double lambda = 0.0) {
  mat fitted = sigmoid(X * beta);
  // compute aux = (fitted - m.y)
  mat aux = fitted;
  for (sp_mat::const_iterator it = y.begin(); it != y.end(); ++it) {
    aux[it.row()] -= m * (*it);
  }
  // return logl gradient = X' * (fitted - m.y)
  mat out = X.t() * aux;
  // add penalizer gradient if necessary
  if (lambda > 0.0) {
    out += lambda * sign(beta);
  }
  // output
  return out;
}


//' @title Fit Logistic Regression with AdaGrad SGD
//' @param beta: numeric vector of parameters of linear model
//' @param X: numeric matrix with predictor variables
//' @param y: numeric vector with response variable
//' @param m: integer controlling the max number of successes in binomial response y
//' @return matrix with one column with the gradient of the negative loglikelihood evaluated at the input
//' @export
// [[Rcpp::export]]
List binom_fit_lazy(const arma::sp_mat & beta0,
                    const arma::sp_mat & Xt,
                    const arma::sp_mat & y,
                    const int m = 1,
                    const double lambda = 0.0,
                    const int minibatch = 1, // don't know how to change for lazy update
                    const int max_epochs = 1e3,
                    const double step_scale = 1.0,
                    const double tol = 1e-6,
                    const double conv_autocor = .06,
                    const bool history = false,
                    const int eval_every = 1,
                    const bool verbose = false) {
  int N = Xt.n_cols, P = Xt.n_rows;
  // Initialize beta
  sp_mat beta = beta0;
  // Initialize lazy update tracker
  mat last_updated(P, 1, fill::zeros); 
  // Initiate adagrad sum of squares that determine the weights
  mat adagrad_ss(P, 1, fill::zeros);
  mat adagrad_wts(P, 1, fill::zeros);
  // Create variables for minibatches indices and data
  int start_row = 0, end_row = minibatch - 1;
  sp_mat Xbatch = Xt.cols(start_row, end_row).t();
  sp_mat ybatch = y.rows(start_row, end_row);
  // adagrad_ss = square(nlogl_binom_grad(beta, Xbatch, ybatch, m, lambda));
  // Initiate variables that detect convergence
  bool converged = false;
  double ll = nlogl_binom(beta, Xbatch, ybatch, m, lambda);
  double ewall = ll, ewall_tmp = ll;
  // Save historic values for sanity checks (not for professional implementation)
  list<mat> hist_wts, hist_updates;
  list<double> hist_nlogl;
  // Main loop
  int epoch_iter = floor(N / minibatch);
  int it;
  for (it = 0; it  < max_epochs * N; it++) {
    // AdaGrad with Lazy Updating 
    // ..compute fitted values only for non-entries 
    double reg = as_scalar(Xbatch * beta);
    double fitted = 1 / (1 + exp(-reg));
    // ..grad terms & adagrad update
    double grad_term = 0.0, debt = 0.0;
    int missing_updates, sgn;
    for (sp_mat::const_iterator j = Xbatch.begin(); j != Xbatch.end(); ++j) {
      // ..compute debt
      missing_updates = it - last_updated[j.col()] - 1;
      sgn = (beta[j.col()] > 0) -  (beta[j.col()] < 0);
      debt = missing_updates * step_scale * lambda * sgn * adagrad_wts[j.col()];
      beta[j.col()] -= debt;
      last_updated[j.col()] = it;
      // ..update beta[j]
      grad_term = (fitted - m * ybatch[j.row()]) * (*j) + lambda * sgn;
      adagrad_ss[j.col()] += pow(grad_term, 2);
      adagrad_wts[j.col()] = 1 / sqrt(adagrad_ss[j.col()] + 1e-8);
      beta[j.col()] +=  - step_scale * adagrad_wts[j.col()] * grad_term;
      if (verbose) {
        Rcout << "col: " << j.col() << " row: " << start_row << " count: " << missing_updates << " grad_term: " 
              << grad_term << " debt: " << debt << " xterm: " << *j << " new betaj: " << beta[j.col()] <<
        " fitted: " << fitted << " reg: " << reg << " ssj: " << adagrad_ss[j.col()] << 
          " wtsj: " << adagrad_wts[j.col()] << " y: " << y[j.row()] << endl ;  
      }
    }
    // Eval (estimated) loglikelihood at newpoint for determining convergence
    if ((it + 1) % eval_every == 0) {
      ll = nlogl_binom(beta, Xbatch, ybatch, m, lambda);
      ewall_tmp = (conv_autocor) * ll + (1 - conv_autocor) * ewall;
      converged = abs((ewall_tmp - ewall) / (ewall + 1e-11)) < tol;
      ewall = ewall_tmp;
      if (history) {
        hist_wts.push_back(adagrad_wts);
        hist_nlogl.push_back(ewall);
        hist_updates.push_back(last_updated);
      }
    }
    // Check convergence or update values
    if (converged) {
      break;
    } 
    else {
      start_row = ((it + 1) % (epoch_iter - 1)) * minibatch;
      end_row = start_row + minibatch - 1;
      Xbatch = Xt.cols(start_row, end_row).t();
      ybatch = y.rows(start_row, end_row);
    }
  }
  return List::create(Named("coefficients") = beta,
                      Named("converged") = converged,
                      Named("iter") = it,
                      Named("epochs") = (it + 0.0)/ N,
                      Named("hist_wts") = hist_wts,
                      Named("hist_nlogl") = hist_nlogl,
                      Named("hist_updates") = hist_updates);
}

