library(Matrix)

X <- readRDS("D:/Github/SDS385/Exercises/Solutions04/Xtoy.RDS")
y <- readRDS("D:/Github/SDS385/Exercises/Solutions04/ytoy.RDS")

nsample <- 10000
psample <- 100

X <- X[1:nsample, 1:psample]
Xt <- t(X)
y <- Matrix(y[1:nsample, ], sparse = TRUE)

m = 1
lambda = 1e-6

beta0 <- Matrix(0, nrow = ncol(X), sparse = TRUE)

res <- binom_fit_lazy(beta0, X, y, lambda = lambda, m = m, max_epochs = 1, verbose = FALSE, eval_every = 1e6, tol = 1e-3)

nlogl_binom(res$coefficients, X, y, m, lambda)
fitted <- round(1 / (1 + exp(- X %*% res$coefficients)))
sum(fitted == y) / nrow(y)

b <- glm.fit(as.matrix(X), as.numeric(y))$coefficients
b[is.na(b)] <- 0

nlogl_binom(Matrix(b, sparse= TRUE), X, y, m, lambda)
fitted <- round(1 / (1 + exp(- X %*% b)))
sum(fitted == y) / nrow(y)


