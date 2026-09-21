import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

from train_final_fusion import MultimodalSurvivalDataset, calculate_concordance_index, cox_partial_likelihood
from model_architectures import SOTAMultimodalFusion

def holdout_randomization_test():
    """
    Executes Pathway D: The Methodological Takedown using the Holdout Randomization Test (HRT).
    Trains the network once, then permutes the images on the test set to build an empirical null.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HRT on {device}...")
    
    if not os.path.exists('aptos_embeddings.npy'):
        print("Required embeddings not found. Run `extract_aptos_embeddings.py` first.")
        return

    dataset = MultimodalSurvivalDataset('aligned_multimodal_cohort.csv', 'stream3_clinical_features.csv')
    
    # Train/Test Split
    n_samples = len(dataset)
    train_idx, test_idx = train_test_split(np.arange(n_samples), test_size=0.2, random_state=42)
    
    train_sub = Subset(dataset, train_idx)
    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    
    model = SOTAMultimodalFusion(geom_dim=6, ehr_dim=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    print("\nTraining Model (1/1)...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        for img_f, geom_f, clin_f, dur, ev in train_loader:
            img_f, geom_f, clin_f = img_f.to(device), geom_f.to(device), clin_f.to(device)
            dur, ev = dur.to(device), ev.to(device)
            
            optimizer.zero_grad()
            risk_preds, _ = model(img_f, geom_f, clin_f, modality_dropout=0.0)
            loss = cox_partial_likelihood(risk_preds.squeeze(-1), dur, ev)
            loss.backward()
            optimizer.step()
            
    model.eval()
    
    # Get Test Data Tensors
    test_img = torch.tensor(dataset.img_tokens[test_idx]).to(device)
    test_geom = torch.tensor(dataset.geom_feats[test_idx]).to(device)
    test_clin = torch.tensor(dataset.clin_feats[test_idx]).to(device)
    test_dur = torch.tensor(dataset.durations[test_idx]).to(device)
    test_ev = torch.tensor(dataset.events[test_idx]).to(device)
    
    with torch.no_grad():
        risk_preds, _ = model(test_img, test_geom, test_clin, modality_dropout=0.0)
    true_c_index = calculate_concordance_index(risk_preds.cpu(), test_dur.cpu(), test_ev.cpu())
    
    print(f"\n-> True Matched Test C-Index: {true_c_index:.4f}")
    
    print("\nExecuting Holdout Randomization Test (100 permutations)...")
    n_perms = 100
    global_null = []
    
    for _ in range(n_perms):
        shuffle_idx = torch.randperm(len(test_idx))
        shuffled_img = test_img[shuffle_idx]
        shuffled_geom = test_geom[shuffle_idx]
        
        with torch.no_grad():
            risk_preds, _ = model(shuffled_img, shuffled_geom, test_clin, modality_dropout=0.0)
        c = calculate_concordance_index(risk_preds.cpu(), test_dur.cpu(), test_ev.cpu())
        global_null.append(c)
        
    global_pval = np.mean(np.array(global_null) >= true_c_index)
    print(f"Global Shuffle (T1) p-value: {global_pval:.4f}")
    
    # Conditional Shuffle (T3)
    # Estimate Propensity Scores on test set for strata
    clf = LogisticRegression()
    clf.fit(dataset.clin_feats[test_idx], np.random.randint(0, 2, len(test_idx))) # Dummy target, we just want a 1D projection score representing covariates
    # Actually, PSM linkage means we can just use the sum of clinical features or a PCA to stratify
    clin_sum = np.sum(dataset.clin_feats[test_idx], axis=1)
    strata = pd.qcut(clin_sum, q=5, labels=False, duplicates='drop')
    
    cond_null = []
    for _ in range(n_perms):
        shuffled_img_c = test_img.clone()
        shuffled_geom_c = test_geom.clone()
        for s in np.unique(strata):
            s_idx = np.where(strata == s)[0]
            s_shuffle = np.random.permutation(s_idx)
            shuffled_img_c[s_idx] = test_img[s_shuffle]
            shuffled_geom_c[s_idx] = test_geom[s_shuffle]
            
        with torch.no_grad():
            risk_preds, _ = model(shuffled_img_c, shuffled_geom_c, test_clin, modality_dropout=0.0)
        c = calculate_concordance_index(risk_preds.cpu(), test_dur.cpu(), test_ev.cpu())
        cond_null.append(c)
        
    cond_pval = np.mean(np.array(cond_null) >= true_c_index)
    print(f"Conditional Shuffle (T3) p-value: {cond_pval:.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    if cond_pval > 0.05:
        print(f"p-value = {cond_pval:.4f} > 0.05. We CANNOT reject the null of Conditional Independence.")
        print("We find NO EVIDENCE that the image provides incremental predictive value beyond the covariates.")
        print("The apparent model performance cannot be statistically distinguished from PSM leakage.")
    else:
        print("Signal detected.")
    print("="*60)

if __name__ == "__main__":
    holdout_randomization_test()
