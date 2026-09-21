import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

def main():
    aligned_csv = 'aligned_multimodal_cohort.csv'
    img_dir = r'd:\Desktop\IPD Zip\IPD Zip\multimodal-stroke-prediction\data\archive\train_images\train_images'
    output_npy = 'aptos_embeddings.npy'
    
    print(f"Loading IDs from {aligned_csv}")
    df = pd.read_csv(aligned_csv)
    aptos_ids = df['APTOS_ID'].values
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pre-trained ResNet-50 on {device}...")
    
    # Load ResNet50 and remove the classification head
    resnet = models.resnet50(pretrained=True)
    # The output of the pooling layer is 2048-dim
    modules = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules).to(device)
    feature_extractor.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    missing_count = 0
    
    print(f"Extracting embeddings for {len(aptos_ids)} images...")
    with torch.no_grad():
        for aid in tqdm(aptos_ids):
            img_path = os.path.join(img_dir, f"{aid}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, f"{aid}.jpeg") # fallback
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    emb = feature_extractor(img_tensor) # (1, 2048, 1, 1)
                    emb = emb.view(2048).cpu().numpy()
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
                    emb = np.zeros(2048, dtype=np.float32)
            else:
                missing_count += 1
                emb = np.zeros(2048, dtype=np.float32)
            
            embeddings.append(emb)
            
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Finished extraction. Shape: {embeddings.shape}")
    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing and zero-padded.")
        
    np.save(output_npy, embeddings)
    print(f"Saved to {output_npy}")

if __name__ == "__main__":
    main()
