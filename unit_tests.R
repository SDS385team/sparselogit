library(Matrix)

X <- readRDS("D:/Github/SDS385/Exercises/Solutions04/Xtoy.RDS")
y <- readRDS("D:/Github/SDS385/Exercises/Solutions04/ytoy.RDS")

X <- X[1:1000, 1:100]
y <- Matrix(y[1:100, ], sparse = TRUE)

m = 1
lambda = .05
res <- binom_fit_lazy(X, y, lambda = lambda, m = m, max_epochs = 100, verbose = FALSE, history = TRUE, eval_every = 1e10, tol = 1e-3)
nlogl_binom(res$coefficients, X, y, m, lambda)
fitted <- round(1 / (1 + exp(- X %*% res$coefficients)))
sum(fitted == y) / nrow(y)

b <- glm.fit(as.matrix(X), as.numeric(y))$coefficients
b[is.na(b)] <- 0
nlogl_binom(Matrix(b, sparse= TRUE), X, y, m, lambda)
fitted <- round(1 / (1 + exp(- X %*% b)))
sum(fitted == y) / nrow(y)


