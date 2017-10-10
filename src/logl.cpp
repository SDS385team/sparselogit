
#include <RcppArmadillo.h>
// [[ Rcpp :: depends ( RcppArmadillo )]]

using namespace Rcpp;
using namespace arma;

//' @title Sigmoid function with truncation
//' @param u: numeric vector of input
//' @param eps: real number of truncation paramaeter if output is close to 0/1 
//' @return sigmoid evaluate at input u with truncation
//' @export
// [[Rcpp::export]]
arma::vec sigmoid(const arma::vec & u, 
                  double eps = 1e-6) {
  int N = u.n_elem;
  arma::vec out(N);
  for (int i = 0; i < N; i++) {
    out[i] = 1 / (1 + exp(-u[i]));
    if (out[i] < eps) {
      out[i] = eps;
    }
    if (out[i] > 1 - eps) {
      out[i] = 1 - eps;
    }
  }
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
double nlogl_binom(const arma::vec & beta, 
                   const arma::mat & X, 
                   const arma::vec & y, 
                   int m) {
  int N = X.n_rows;
  double out = 0.0;
  arma::vec fitted(N);
  //
  fitted = sigmoid(X * beta);
  for (int i = 0; i < N; i++) {
    out += y[i] * log(fitted[i]) + (m - y[i]) * log(1 - fitted[i]);
  }
  return out;
}
