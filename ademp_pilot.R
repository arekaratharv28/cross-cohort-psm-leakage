# ADEMP Simulation Pilot: Tier 1 DGM and T1-T6 Tests
# ---------------------------------------------------------
# Required packages: survival, MatchIt, pROC
# Install via: install.packages(c("survival", "MatchIt", "pROC"))

library(survival)
library(MatchIt)

# ==========================================
# 1. DATA GENERATING MECHANISM (Tier 1)
# ==========================================
generate_population <- function(n = 3000, rho = sqrt(0.5), gamma = 0.5) {
  # C: Covariates
  Age <- rnorm(n, 60, 10)
  Sex <- rbinom(n, 1, 0.5)
  
  # Age -> HTN -> DM dependence
  logit_HTN <- -5 + 0.08 * Age
  HTN <- rbinom(n, 1, plogis(logit_HTN))
  
  logit_DM <- -4 + 0.05 * Age + 1.5 * HTN
  DM <- rbinom(n, 1, plogis(logit_DM))
  
  C_matrix <- cbind(Age, Sex, HTN, DM)
  
  # U: Latent retinal state
  # z(C) is a linear combination of C
  z_C <- scale(0.3 * scale(Age) + 0.2 * Sex + 0.6 * HTN + 0.5 * DM)
  U <- rho * z_C + sqrt(1 - rho^2) * rnorm(n)
  
  # I: Image embeddings (d = 16)
  d <- 16
  I <- matrix(rnorm(n * d), nrow = n, ncol = d)
  # Plant U into the first few dimensions of I
  I[, 1] <- I[, 1] + 2 * U
  I[, 2] <- I[, 2] + 1.5 * U
  
  # T, delta: Survival outcome (Weibull baseline)
  # Hazard: h(t) = exp(beta'C + gamma*U)
  linear_predictor <- 0.05 * scale(Age) + 0.1 * Sex + 0.4 * HTN + 0.3 * DM + gamma * U
  
  # Inverse transform sampling for Weibull survival time
  lambda <- 0.001
  v <- 1.5
  u <- runif(n)
  T_surv <- (-log(u) / (lambda * exp(linear_predictor)))^(1/v)
  
  # Censoring
  Cens <- rexp(n, 0.05)
  time <- pmin(T_surv, Cens)
  status <- ifelse(T_surv <= Cens, 1, 0)
  
  data.frame(ID = 1:n, Age, Sex, HTN, DM, U, time, status, I)
}

# ==========================================
# 2. DESIGNS (P vs L0)
# ==========================================
get_design_data <- function(pop, design = "P") {
  img_cols <- paste0("X", 1:16)
  
  if (design == "P") {
    # P: Truly paired
    return(pop)
  } else if (design == "L0") {
    # L0: Linked, same population (split and PSM match)
    # Split population in half
    half <- floor(nrow(pop) / 2)
    cohort_A <- pop[1:half, ]           # Keep Image, discard Outcome
    cohort_B <- pop[(half + 1):nrow(pop), ] # Keep Outcome, discard Image
    
    # PSM Matching on C
    # We combine them, create a dummy treatment, and match
    cohort_A$treat <- 1
    cohort_B$treat <- 0
    pool <- rbind(cohort_A, cohort_B)
    
    m.out <- matchit(treat ~ Age + Sex + HTN + DM, data = pool, method = "nearest", ratio = 1)
    matched_data <- match.data(m.out)
    
    # Construct the linked dataset
    treated <- matched_data[matched_data$treat == 1, ]
    control <- matched_data[matched_data$treat == 0, ]
    
    # Order them by subclass so they align perfectly
    treated <- treated[order(treated$subclass), ]
    control <- control[order(control$subclass), ]
    
    # The L0 dataset takes Outcomes (time, status) from cohort B (control)
    # and Images (X1..X16) from cohort A (treated)
    # Covariates from cohort B (as that's where the target belongs)
    linked <- control[, c("ID", "Age", "Sex", "HTN", "DM", "time", "status")]
    linked[, img_cols] <- treated[, img_cols]
    
    return(linked)
  }
}

# ==========================================
# 3. METHODS & TESTS (T1 - T6)
# ==========================================
run_tests <- function(data) {
  img_cols <- paste0("X", 1:16)
  
  # Base models
  form_C <- as.formula("Surv(time, status) ~ Age + Sex + HTN + DM")
  form_CI <- as.formula(paste("Surv(time, status) ~ Age + Sex + HTN + DM +", paste(img_cols, collapse = "+")))
  
  mod_C <- coxph(form_C, data = data)
  mod_CI <- coxph(form_CI, data = data)
  
  c_C <- summary(mod_C)$concordance["C"]
  c_CI <- summary(mod_CI)$concordance["C"]
  
  results <- list()
  results$C_baseline <- c_C
  results$C_fusion <- c_CI
  
  # T1: The original naive global shuffle (flag if C_fusion stays above 0.5)
  data_T1 <- data
  data_T1[, img_cols] <- data_T1[sample(nrow(data_T1)), img_cols]
  mod_T1 <- coxph(form_CI, data = data_T1)
  c_T1 <- summary(mod_T1)$concordance["C"]
  # Flag 1 if performance drops by less than 0.03 (indicating leakage / no signal)
  results$T1_flag <- ifelse((c_CI - c_T1) < 0.03, 1, 0)
  
  # T2: Global permutation test (B = 100 for pilot speed)
  B <- 50
  deltas_T2 <- numeric(B)
  for (i in 1:B) {
    shuffled <- data
    shuffled[, img_cols] <- shuffled[sample(nrow(shuffled)), img_cols]
    m_shuff <- coxph(form_CI, data = shuffled)
    deltas_T2[i] <- c_CI - summary(m_shuff)$concordance["C"]
  }
  results$T2_pval <- mean(deltas_T2 >= (c_CI - c_C))
  
  # T3: Conditional permutation test (within PS strata)
  # Compute PS for stratification (using covariates predicting an arbitrary binary event)
  # Since it's within-cohort, we group by Age deciles and HTN status as a proxy for PS strata
  strata <- paste(ntile(data$Age, 5), data$HTN)
  deltas_T3 <- numeric(B)
  for (i in 1:B) {
    shuffled <- data
    for (s in unique(strata)) {
      idx <- which(strata == s)
      shuffled[idx, img_cols] <- shuffled[sample(idx), img_cols]
    }
    m_shuff <- coxph(form_CI, data = shuffled)
    deltas_T3[i] <- c_CI - summary(m_shuff)$concordance["C"]
  }
  results$T3_pval <- mean(deltas_T3 >= (c_CI - c_C))
  
  # T4: Nested Cox LRT (C vs C + I)
  lrt <- anova(mod_C, mod_CI)
  results$T4_pval <- lrt$`P(>|Chi|)`[2]
  
  # T6: Generalised Covariance Measure (Simplified residual test)
  # Regress Image PC1 on C, Regress Martingale Residuals on C
  pca <- prcomp(data[, img_cols], scale. = TRUE)
  I_pc1 <- pca$x[, 1]
  
  res_I <- residuals(lm(I_pc1 ~ Age + Sex + HTN + DM, data = data))
  res_M <- residuals(mod_C, type = "martingale")
  
  # Test covariance of residuals
  cov_test <- cor.test(res_I, res_M)
  results$T6_pval <- cov_test$p.value
  
  return(results)
}

# Helper ntile
ntile <- function(x, n) {
  floor((n * (rank(x, ties.method = "first") - 1)) / length(x)) + 1
}

# ==========================================
# 4. PILOT RUN (100 Repetitions)
# ==========================================
run_pilot <- function(reps = 100, design = "P", gamma = 0.5) {
  cat(sprintf("Running ADEMP Pilot (Reps=%d, Design=%s, Gamma=%.1f)\n", reps, design, gamma))
  
  t1_flags <- 0
  t4_rejections <- 0
  t6_rejections <- 0
  
  pb <- txtProgressBar(min = 0, max = reps, style = 3)
  for (r in 1:reps) {
    pop <- generate_population(n = 2000, gamma = gamma)
    data <- get_design_data(pop, design)
    
    res <- run_tests(data)
    
    t1_flags <- t1_flags + res$T1_flag
    t4_rejections <- t4_rejections + (res$T4_pval < 0.05)
    t6_rejections <- t6_rejections + (res$T6_pval < 0.05)
    
    setTxtProgressBar(pb, r)
  }
  close(pb)
  
  cat("\n--- Pilot Results ---\n")
  cat(sprintf("T1 (Naive Global Shuffle) 'No Signal' Flag Rate: %.1f%%\n", (t1_flags/reps)*100))
  cat(sprintf("T4 (Nested LRT) Type-I/Power Rejection Rate:    %.1f%%\n", (t4_rejections/reps)*100))
  cat(sprintf("T6 (GCM-Proxy) Type-I/Power Rejection Rate:     %.1f%%\n", (t6_rejections/reps)*100))
}

# Example Executions
# run_pilot(reps=50, design="P", gamma=0.5)  # Truly Paired (Should have high power for T4, T6)
# run_pilot(reps=50, design="L0", gamma=0.5) # PSM Linked (Should collapse to ~5% false positive rate)

