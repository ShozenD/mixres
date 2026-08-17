#!/usr/bin/env Rscript
# Estimate the fine-resolution Poisson rate from aggregate counts using
# ungroup::pclm(), and compute error metrics (MSE, interval score, 95%
# coverage) against the known true rate function.
#
# pclm()'s default confidence interval (fit$ci, computed via the internal
# pclm.confidence() -> pclm.confidence.dx() branch) applies a life-table
# "dx" (death-count) transform intended for monotonically-declining
# mortality data. On our non-monotonic simulated rate function it produces
# invalid, sometimes-inverted (upper < lower) bounds -- verified directly:
# 62/100 grid points came out inverted for a seed=3, bin_width=10 fit. We
# instead call pclm.confidence's "mx" (rate-scale, log-normal) branch,
# ungroup:::pclm.confidence.mx(), directly on the same fit and standard
# errors pclm() already computed -- the same computation pclm() would use
# internally for a rate-with-exposure model -- without altering the fitted
# values themselves.

library(ungroup)

args <- commandArgs(trailingOnly = TRUE)
seed <- if (length(args) >= 1) as.integer(args[1]) else 1L
bin_width <- if (length(args) >= 2) as.integer(args[2]) else 5L

data_path <- file.path("data", "sim", sprintf("sim_poisson_1d_%d_bw%d.csv", seed, bin_width))
fit_path <- file.path("data", "sim", sprintf("sim_poisson_1d_%d_bw%d_ungroup_fit.csv", seed, bin_width))

df <- read.csv(data_path)

bins <- df[!duplicated(df$bin_id), c("bin_left", "bin_right", "y")]
bins <- bins[order(bins$bin_left), ]

x <- bins$bin_left
y <- bins$y
nlast <- bins$bin_right[nrow(bins)] - bins$bin_left[nrow(bins)] + 1

fit <- pclm(x, y, nlast = nlast)
fitted_vals <- as.numeric(fit$fitted)

alpha <- 0.05
se <- as.numeric(fit$goodness.of.fit$standard.errors)
ci <- ungroup:::pclm.confidence.mx(list(
  fit = fitted_vals, SE = se, qn = qnorm(1 - alpha / 2), type = "1D"
))
lower <- ci$lower
upper <- ci$upper

truth <- df[order(df$x), c("x", "true_lambda")]
true_lambda <- truth$true_lambda

mse <- mean((fitted_vals - true_lambda)^2)

# Interval score (Gneiting & Raftery, 2007) for a (1 - alpha) interval:
# rewards a narrow interval, penalized when the true value falls outside it.
interval_score <- mean(
  (upper - lower) +
    (2 / alpha) * (lower - true_lambda) * (true_lambda < lower) +
    (2 / alpha) * (true_lambda - upper) * (true_lambda > upper)
)

coverage <- mean(true_lambda >= lower & true_lambda <= upper)

cat(sprintf("MSE: %.4f\n", mse))
cat(sprintf("Interval score (95%%): %.4f\n", interval_score))
cat(sprintf("95%% coverage probability: %.4f\n", coverage))

out <- data.frame(x = truth$x, fitted = fitted_vals, lower = lower, upper = upper)
write.csv(out, fit_path, row.names = FALSE)
cat(sprintf("Saved fitted values to %s\n", fit_path))
