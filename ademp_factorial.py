import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import chi2, pearsonr
from tqdm import tqdm
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA GENERATING MECHANISM
# ==========================================
def generate_population(n, rho, gamma, d=16):
    Age = np.random.normal(60, 10, n)
    Sex = np.random.binomial(1, 0.5, n)
    
    logit_HTN = -5 + 0.08 * Age
    HTN = np.random.binomial(1, 1 / (1 + np.exp(-logit_HTN)), n)
    
    logit_DM = -4 + 0.05 * Age + 1.5 * HTN
    DM = np.random.binomial(1, 1 / (1 + np.exp(-logit_DM)), n)
    
    z_C = 0.3 * ((Age - np.mean(Age)) / np.std(Age)) + 0.2 * Sex + 0.6 * HTN + 0.5 * DM
    z_C_scaled = (z_C - np.mean(z_C)) / np.std(z_C)
    U = rho * z_C_scaled + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
    
    I = np.random.normal(0, 1, (n, d))
    I[:, 0] += 2 * U
    I[:, 1] += 1.5 * U
    
    linear_predictor = 0.05 * ((Age - np.mean(Age)) / np.std(Age)) + 0.1 * Sex + 0.4 * HTN + 0.3 * DM + gamma * U
    
    lambda_ = 0.001
    v = 1.5
    u = np.random.uniform(0, 1, n)
    T_surv = (-np.log(u) / (lambda_ * np.exp(linear_predictor))) ** (1 / v)
    
    Cens = np.random.exponential(1/0.05, n)
    time = np.minimum(T_surv, Cens)
    status = (T_surv <= Cens).astype(int)
    
    df = pd.DataFrame({'ID': np.arange(1, n + 1), 'Age': Age, 'Sex': Sex, 'HTN': HTN, 'DM': DM, 'U': U, 'time': time, 'status': status})
    for i in range(d):
        df[f'X{i+1}'] = I[:, i]
        
    return df

def get_design_data(pop, design="P"):
    if design == "P":
        return pop.copy()
    
    elif design in ["L0", "L2"]:
        half = len(pop) // 2
        cohort_A = pop.iloc[:half].copy()
        cohort_B = pop.iloc[half:].copy()
        
        cohort_A['treat'] = 1
        cohort_B['treat'] = 0
        pool = pd.concat([cohort_A, cohort_B], ignore_index=True)
        
        clf = LogisticRegression(solver='lbfgs')
        match_cols = ['Age', 'Sex', 'HTN', 'DM'] if design == "L0" else ['Age', 'HTN', 'DM']
        
        clf.fit(pool[match_cols], pool['treat'])
        pool['ps'] = clf.predict_proba(pool[match_cols])[:, 1]
        
        treated = pool[pool['treat'] == 1].reset_index(drop=True)
        control = pool[pool['treat'] == 0].reset_index(drop=True)
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[['ps']])
        distances, indices = nn.kneighbors(treated[['ps']])
        
        matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
        
        linked = matched_control[['ID', 'Age', 'Sex', 'HTN', 'DM', 'time', 'status', 'ps']].copy()
        img_cols = [f'X{i+1}' for i in range(16)]
        for col in img_cols:
            linked[col] = treated[col].values
            
        return linked

def run_tests(data):
    img_cols = [f'X{i+1}' for i in range(16)]
    c_cols = ['Age', 'Sex', 'HTN', 'DM', 'time', 'status']
    ci_cols = c_cols + img_cols
    
    c_data = data[c_cols].copy()
    ci_data = data[ci_cols].copy()
    
    for col in ['Age', 'Sex', 'HTN', 'DM']:
        ci_data[col] += np.random.normal(0, 1e-5, len(ci_data))
        c_data[col] += np.random.normal(0, 1e-5, len(c_data))
    
    try:
        mod_C = CoxPHFitter(penalizer=1e-4)
        mod_C.fit(c_data, duration_col='time', event_col='status')
        
        mod_CI = CoxPHFitter(penalizer=1e-4)
        mod_CI.fit(ci_data, duration_col='time', event_col='status')
        c_CI = mod_CI.concordance_index_
    except:
        return None
        
    results = {}
    
    ll_C = mod_C.log_likelihood_
    ll_CI = mod_CI.log_likelihood_
    lrt_stat = -2 * (ll_C - ll_CI)
    results['T4_pval'] = chi2.sf(lrt_stat, df=16)

    try:
        resid_M = mod_C.compute_residuals(c_data, 'martingale')['martingale'].values
        pca = PCA(n_components=1)
        I_pc1 = pca.fit_transform(ci_data[img_cols].values).flatten()
        
        lin = LinearRegression()
        lin.fit(c_data[['Age', 'Sex', 'HTN', 'DM']], I_pc1)
        resid_I = I_pc1 - lin.predict(c_data[['Age', 'Sex', 'HTN', 'DM']])
        
        _, gcm_pval = pearsonr(resid_I, resid_M)
        results['T6_pval'] = gcm_pval
    except:
        results['T6_pval'] = 1.0

    n_perms = 25
    try:
        data_T3 = ci_data.copy()
        data_T3['Age_q'] = pd.qcut(data_T3['Age'], q=5, labels=False, duplicates='drop')
        data_T3['stratum'] = data_T3['Age_q'].astype(str) + "_" + data_T3['HTN'].astype(str)
        
        c_C = mod_C.concordance_index_
        null_c = []
        for _ in range(n_perms):
            data_T3_perm = data_T3.copy()
            for stratum in data_T3_perm['stratum'].unique():
                idx = data_T3_perm.index[data_T3_perm['stratum'] == stratum].tolist()
                shuffled_idx = np.random.permutation(idx)
                data_T3_perm.loc[idx, img_cols] = data_T3_perm.loc[shuffled_idx, img_cols].values
                
            mod_T3 = CoxPHFitter(penalizer=1e-4)
            mod_T3.fit(data_T3_perm.drop(columns=['Age_q', 'stratum']), duration_col='time', event_col='status')
            null_c.append(mod_T3.concordance_index_)
            
        results['T3_pval'] = np.mean(np.array(null_c) >= c_CI)
    except:
        results['T3_pval'] = 1.0

    return results

def process_scenario(args):
    design, gamma, rho, n, reps = args
    t3_rejs, t4_rejs, t6_rejs, successes = 0, 0, 0, 0
    
    np.random.seed(int((gamma + rho + len(design)) * 100000) % 4294967295)
    
    for _ in range(reps):
        pop = generate_population(n=n, rho=rho, gamma=gamma)
        data = get_design_data(pop, design)
        res = run_tests(data)
        
        if res is not None:
            successes += 1
            t3_rejs += int(res['T3_pval'] < 0.05)
            t4_rejs += int(res['T4_pval'] < 0.05)
            t6_rejs += int(res['T6_pval'] < 0.05)
            
    if successes > 0:
        return {
            'Design': design,
            'Planted_Gamma': gamma,
            'Rho_sq': round(rho**2, 1),
            'N': n,
            'Success_Fits': successes,
            'T3_Power': round(t3_rejs/successes, 3),
            'T4_Power': round(t4_rejs/successes, 3),
            'T6_Power': round(t6_rejs/successes, 3)
        }
    return None

def run_full_factorial_grid():
    designs = ["P", "L0", "L2"]
    gammas = [0.0, 0.5, 1.0]
    rhos = [np.sqrt(0.2), np.sqrt(0.5), np.sqrt(0.8)]
    sample_sizes = [1000] 
    
    reps_per_scenario = 100
    
    grid = list(itertools.product(designs, gammas, rhos, sample_sizes))
    args_list = [(d, g, r, n, reps_per_scenario) for d, g, r, n in grid]
    
    print(f"Executing Full ADEMP Factorial Grid: {len(grid)} Scenarios, {reps_per_scenario} Reps Each.")
    print("Using joblib loky backend for safe Windows multiprocessing...")
    
    from joblib import Parallel, delayed
    
    results_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_scenario)(args) for args in tqdm(args_list, total=len(grid))
    )
    
    # Filter out Nones
    results_list = [r for r in results_list if r is not None]
            
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(['Design', 'Planted_Gamma', 'Rho_sq'], ascending=[False, True, True])
    df_results.to_csv("ADEMP_Factorial_Results.csv", index=False)
    print("\nGrid complete. Results saved to 'ADEMP_Factorial_Results.csv'")

if __name__ == "__main__":
    run_full_factorial_grid()
