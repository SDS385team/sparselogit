library(Matrix)
library(readsvm)
# library(microbenchmark)

# res <- read_sparse_svm("D:/Datasets/full_url_svmlight.svm", return_tranpose = FALSE)
# saveRDS(res, "D:/Datasets/full_url_svmlight.RDS")
res <- readRDS("D:/Datasets/full_url_svmlight.RDS")
nsample <- ncol(res$features)
# nsample <- 2
psample <- nrow(res$features)

# Xt <- res$features[1:psample, 1:nsample, drop = FALSE]
y <- as.integer(res$response[1:nsample] == 1)
table(y)

beta0 <- numeric(psample)

system.time({
  mod <- sparse_logit(beta0, res$features, y, step_scale = 2, lambda = 1e-8, maxepochs = 100)
})

mod$alpha
summary(mod$coefficients)
hist(sample(mod$fitted))

fitted_rounded <- round(mod$fitted)
table(y, fitted_rounded)
sum(y == fitted_rounded) / length(y)

plot(sort(sample(mod$coefficients, 1e4)), type = "l")

