import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import pearsonr

# ==========================================
# 1. DATA GENERATING MECHANISM (Tier 1)
# ==========================================
def generate_population(n=3000, rho=np.sqrt(0.5), gamma=0.5):
    # C: Covariates
    Age = np.random.normal(60, 10, n)
    Sex = np.random.binomial(1, 0.5, n)
    
    # Age -> HTN -> DM dependence
    logit_HTN = -5 + 0.08 * Age
    prob_HTN = 1 / (1 + np.exp(-logit_HTN))
    HTN = np.random.binomial(1, prob_HTN, n)
    
    logit_DM = -4 + 0.05 * Age + 1.5 * HTN
    prob_DM = 1 / (1 + np.exp(-logit_DM))
    DM = np.random.binomial(1, prob_DM, n)
    
    # U: Latent retinal state
    z_C = 0.3 * ((Age - np.mean(Age)) / np.std(Age)) + 0.2 * Sex + 0.6 * HTN + 0.5 * DM
    z_C_scaled = (z_C - np.mean(z_C)) / np.std(z_C)
    U = rho * z_C_scaled + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
    
    # I: Image embeddings (d = 16)
    d = 16
    I = np.random.normal(0, 1, (n, d))
    I[:, 0] += 2 * U
    I[:, 1] += 1.5 * U
    
    # T, delta: Survival outcome (Weibull baseline)
    linear_predictor = 0.05 * ((Age - np.mean(Age)) / np.std(Age)) + 0.1 * Sex + 0.4 * HTN + 0.3 * DM + gamma * U
    
    lambda_ = 0.001
    v = 1.5
    u = np.random.uniform(0, 1, n)
    T_surv = (-np.log(u) / (lambda_ * np.exp(linear_predictor))) ** (1 / v)
    
    # Censoring
    Cens = np.random.exponential(1/0.05, n)
    time = np.minimum(T_surv, Cens)
    status = (T_surv <= Cens).astype(int)
    
    df = pd.DataFrame({
        'ID': np.arange(1, n + 1),
        'Age': Age,
        'Sex': Sex,
        'HTN': HTN,
        'DM': DM,
        'U': U,
        'time': time,
        'status': status
    })
    
    for i in range(d):
        df[f'X{i+1}'] = I[:, i]
        
    return df

# ==========================================
# 2. DESIGNS (P vs L0)
# ==========================================
def get_design_data(pop, design="P"):
    if design == "P":
        return pop.copy()
    
    elif design == "L0":
        half = len(pop) // 2
        cohort_A = pop.iloc[:half].copy()           # Keep Image
        cohort_B = pop.iloc[half:].copy()           # Keep Outcome
        
        cohort_A['treat'] = 1
        cohort_B['treat'] = 0
        pool = pd.concat([cohort_A, cohort_B], ignore_index=True)
        
        # Propensity Score Matching
        clf = LogisticRegression(solver='lbfgs')
        X = pool[['Age', 'Sex', 'HTN', 'DM']]
        clf.fit(X, pool['treat'])
        pool['ps'] = clf.predict_proba(X)[:, 1]
        
        treated = pool[pool['treat'] == 1].reset_index(drop=True)
        control = pool[pool['treat'] == 0].reset_index(drop=True)
        
        # 1:1 Nearest Neighbor on Propensity Score
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[['ps']])
        distances, indices = nn.kneighbors(treated[['ps']])
        
        matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
        
        # Construct linked dataset
        linked = matched_control[['ID', 'Age', 'Sex', 'HTN', 'DM', 'time', 'status']].copy()
        img_cols = [f'X{i+1}' for i in range(16)]
        for col in img_cols:
            linked[col] = treated[col].values
            
        return linked

# ==========================================
# 3. METHODS & TESTS (T1, T4, T6)
# ==========================================
def run_tests(data):
    img_cols = [f'X{i+1}' for i in range(16)]
    
    results = {}
    
    # 1. Base Models
    c_cols = ['Age', 'Sex', 'HTN', 'DM', 'time', 'status']
    ci_cols = c_cols + img_cols
    
    c_data = data[c_cols].copy()
    ci_data = data[ci_cols].copy()
    
    # Add small noise to identical covariates to prevent Lifelines convergence issues on simulated data
    for col in ['Age', 'Sex', 'HTN', 'DM']:
        ci_data[col] += np.random.normal(0, 1e-5, len(ci_data))
        c_data[col] += np.random.normal(0, 1e-5, len(c_data))
    
    try:
        mod_C = CoxPHFitter(penalizer=0.1) # Ridge penalty to ensure convergence
        mod_C.fit(c_data, duration_col='time', event_col='status')
        c_C = mod_C.concordance_index_
        
        mod_CI = CoxPHFitter(penalizer=0.1)
        mod_CI.fit(ci_data, duration_col='time', event_col='status')
        c_CI = mod_CI.concordance_index_
    except Exception as e:
        return None # Return None if fit fails
        
    # T1: Naive Global Shuffle
    data_T1 = ci_data.copy()
    data_T1[img_cols] = data_T1[img_cols].sample(frac=1).values
    
    try:
        mod_T1 = CoxPHFitter(penalizer=0.1)
        mod_T1.fit(data_T1, duration_col='time', event_col='status')
        c_T1 = mod_T1.concordance_index_
        results['T1_flag'] = 1 if (c_CI - c_T1) < 0.03 else 0
    except:
        results['T1_flag'] = 0
        
    # T4: Likelihood Ratio Test Approximation
    # Since lifelines doesn't have a direct nested LRT function for penalized models that returns a p-value easily,
    # we use the difference in log-likelihoods
    ll_C = mod_C.log_likelihood_
    ll_CI = mod_CI.log_likelihood_
    lrt_stat = -2 * (ll_C - ll_CI)
    from scipy.stats import chi2
    results['T4_pval'] = chi2.sf(lrt_stat, df=16)
    
    # T6: Generalised Covariance Measure (Residual Proxy)
    pca = PCA(n_components=1)
    I_pc1 = pca.fit_transform(data[img_cols])[:, 0]
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(data[['Age', 'Sex', 'HTN', 'DM']], I_pc1)
    res_I = I_pc1 - lr.predict(data[['Age', 'Sex', 'HTN', 'DM']])
    
    res_M = mod_C.compute_residuals(c_data, kind='martingale')['martingale'].values
    
    r, p_val = pearsonr(res_I, res_M)
    results['T6_pval'] = p_val
    
    return results

# ==========================================
# 4. PILOT RUN
# ==========================================
def run_pilot(reps=100, design="P", gamma=0.5):
    print(f"\nRunning ADEMP Pilot (Reps={reps}, Design={design}, Gamma={gamma})")
    
    t1_flags = 0
    t4_rejections = 0
    t6_rejections = 0
    successful_reps = 0
    
    for _ in tqdm(range(reps)):
        pop = generate_population(n=2000, gamma=gamma)
        data = get_design_data(pop, design)
        
        res = run_tests(data)
        if res is not None:
            t1_flags += res['T1_flag']
            t4_rejections += int(res['T4_pval'] < 0.05)
            t6_rejections += int(res['T6_pval'] < 0.05)
            successful_reps += 1
            
    if successful_reps == 0:
        print("All models failed to converge.")
        return
        
    print("\n--- Pilot Results ---")
    print(f"Successful Fits: {successful_reps}/{reps}")
    print(f"T1 (Naive Global Shuffle) 'No Signal' Flag Rate: {(t1_flags/successful_reps)*100:.1f}%")
    print(f"T4 (Nested LRT) Type-I/Power Rejection Rate:    {(t4_rejections/successful_reps)*100:.1f}%")
    print(f"T6 (GCM-Proxy) Type-I/Power Rejection Rate:     {(t6_rejections/successful_reps)*100:.1f}%")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n[EXPERIMENT 1] TRUE PAIRING (Ground Truth Positive Control)")
    print("Expectation: T4 and T6 should reject the null hypothesis (high power) because genuine signal exists.")
    run_pilot(reps=30, design="P", gamma=1.0)
    
    print("\n[EXPERIMENT 2] L0 PSM-LINKED DATA (Leakage Proof)")
    print("Expectation: T4 and T6 rejections should collapse to ~5% (Type-I error rate) because the image signal is fake.")
    run_pilot(reps=30, design="L0", gamma=1.0)

