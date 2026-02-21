import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

with open('temp_import.py', 'w') as f:
    f.write("""

def analyze_colors(image_rgb, step=5):
    sampled_img = image_rgb[::step, ::step]
    r, g, b = cv2.split(sampled_img)
    return r, g, b

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast_enhanced, (7,7), 0)
    _, binary = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def segment_cells_improved(image, binary):
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    
    if np.max(dist_transform) > 0:
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_norm = np.uint8(dist_norm)
        _, sure_fg = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        sure_fg = cleaned.copy()
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    gradient = cv2.morphologyEx(gray_enhanced, cv2.MORPH_GRADIENT, kernel)
    markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)
    
    return markers

def count_and_filter(image, markers, min_size=100, max_size=600):
    counts = 0
    areas = []
    for label in np.unique(markers):
        if label in [0, -1]:
            continue
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
            if min_size <= area <= max_size:
                counts += 1
    return counts, areas

def process_single_image(image_path, step=5):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None, None
    filename = os.path.basename(image_path)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = analyze_colors(img_rgb, step)
        binary = preprocess_image(img)
        markers = segment_cells_improved(img, binary)
        cell_count, areas = count_and_filter(img, markers)
        color_stats = {
            'filename': filename,
            'Red': np.mean(r),
            'Green': np.mean(g),
            'Blue': np.mean(b),
            'R/G': np.mean(r) / (np.mean(g) + 0.001),
            'R/B': np.mean(r) / (np.mean(b) + 0.001),
            'G/B': np.mean(g) / (np.mean(b) + 0.001),
            'pixels_sampled': r.size
        }
        cell_stats = {
            'Cell Count': cell_count,
            'Mean Area': np.mean(areas) if areas else 0,
            'Median Area': np.median(areas) if areas else 0,
            'Min Area': min(areas) if areas else 0,
            'Max Area': max(areas) if areas else 0,
            'Areas': areas
        }
        return color_stats, cell_stats, r, g, b
    except Exception:
        return None, None, None, None, None
""")

import temp_import
process_single_image = temp_import.process_single_image

class DataNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        return self
    
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        normalized = (data - self.mean) / self.std
        return np.clip(normalized, -5, 5)
    
    def inverse_transform(self, data):
        return data * self.std + self.mean

class SafePhenotypeLoss(nn.Module):
    def __init__(self, color_weight=1.0, morph_weight=0.8):
        super().__init__()
        self.color_weight = color_weight
        self.morph_weight = morph_weight
        self.clip_value = 5.0
        
    def forward(self, pred_color, pred_morph, target_color, target_morph):
        pred_color = torch.clamp(pred_color, -self.clip_value, self.clip_value)
        pred_morph = torch.clamp(pred_morph, -self.clip_value, self.clip_value)
        target_color = torch.clamp(target_color, -self.clip_value, self.clip_value)
        target_morph = torch.clamp(target_morph, -self.clip_value, self.clip_value)
        
        loss_color = F.huber_loss(pred_color, target_color, reduction='mean', delta=1.0)
        loss_morph = F.huber_loss(pred_morph, target_morph, reduction='mean', delta=1.0)
        total_loss = self.color_weight * loss_color + self.morph_weight * loss_morph
        
        return {'total': total_loss, 'color': loss_color, 'morph': loss_morph}

class SafeChlorellaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        params = list(self.backbone.parameters())
        freeze_until = int(len(params) * 0.8)
        for i, param in enumerate(params):
            param.requires_grad = (i >= freeze_until)
        
        self.backbone.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(512)
        
        self.color_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 6)
        )
        self.morph_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 7)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.bn(self.backbone(x))
        color_pred = torch.tanh(self.color_head(features)) * 3
        morph_pred = torch.tanh(self.morph_head(features)) * 3
        return color_pred, morph_pred

class ChlorellaDataset(Dataset):
    def __init__(self, df, images_dir=".", transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        
        img = self.transform(img)
        color_target = torch.FloatTensor([row[f'{c}_norm'] for c in ['Red', 'Green', 'Blue', 'R/G', 'R/B', 'G/B']])
        morph_cols = ['Cell_Count', 'Mean_Area', 'Median_Area', 'Min_Area', 'Max_Area', 'Std_Area', 'Total_Area']
        morph_target = torch.FloatTensor([row[f'{c}_norm'] for c in morph_cols])
        return img, color_target, morph_target, row['filename']

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    
    params = {
        'lr': 1e-5, 'weight_decay': 1e-4, 'max_grad_norm': 1.0,
        'patience': 20, 'min_lr': 1e-7, 'scheduler_factor': 0.5,
        'scheduler_patience': 5, 'num_epochs': 100, 'batch_size': 16
    }
    
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.bmp', '*.tif']:
        image_paths.extend(glob.glob(os.path.join(config['paths']['images_dir'], ext)))
    
    data_rows = []
    for img_path in tqdm(image_paths[:1050], desc="Processing"):
        color_stats, cell_stats, _, _, _ = process_single_image(img_path)
        if color_stats and cell_stats:
            areas = cell_stats.get('Areas', [0])
            row = {
                'filename': os.path.basename(img_path),
                'Red': color_stats['Red'], 'Green': color_stats['Green'], 'Blue': color_stats['Blue'],
                'R/G': color_stats['R/G'], 'R/B': color_stats['R/B'], 'G/B': color_stats['G/B'],
                'Cell_Count': cell_stats['Cell Count'], 'Mean_Area': cell_stats['Mean Area'],
                'Median_Area': cell_stats['Median Area'], 'Min_Area': cell_stats['Min Area'],
                'Max_Area': cell_stats['Max Area'], 'Std_Area': np.std(areas), 'Total_Area': sum(areas)
            }
            data_rows.append(row)

    df_raw = pd.DataFrame(data_rows)
    df_normalized = df_raw.copy()
    
    color_cols = ['Red', 'Green', 'Blue', 'R/G', 'R/B', 'G/B']
    morph_cols = ['Cell_Count', 'Mean_Area', 'Median_Area', 'Min_Area', 'Max_Area', 'Std_Area', 'Total_Area']
    
    c_norm = DataNormalizer().fit(df_raw[color_cols])
    m_norm = DataNormalizer().fit(df_raw[morph_cols])
    
    df_normalized[[f'{c}_norm' for c in color_cols]] = c_norm.transform(df_raw[color_cols])
    df_normalized[[f'{c}_norm' for c in morph_cols]] = m_norm.transform(df_raw[morph_cols])
    
    train_df, val_df = train_test_split(df_normalized, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(ChlorellaDataset(train_df, config['paths']['images_dir']), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(ChlorellaDataset(val_df, config['paths']['images_dir']), batch_size=params['batch_size'])
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = SafeChlorellaModel().to(device)
    criterion = SafePhenotypeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['scheduler_factor'], patience=params['scheduler_patience'], min_lr=params['min_lr'])
    
    for epoch in range(params['num_epochs']):
        model.train()
        for imgs, c_target, m_target, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs, c_target, m_target = imgs.to(device), c_target.to(device), m_target.to(device)
            optimizer.zero_grad()
            c_pred, m_pred = model(imgs)
            loss = criterion(c_pred, m_pred, c_target, m_target)['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, c_target, m_target, _ in val_loader:
                imgs, c_target, m_target = imgs.to(device), c_target.to(device), m_target.to(device)
                c_pred, m_pred = model(imgs)
                val_loss += criterion(c_pred, m_pred, c_target, m_target)['total'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def main():

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_color'].append(train_color_loss / len(train_loader))
    history['train_morph'].append(train_morph_loss / len(train_loader))
    history['val_color'].append(val_color_loss / len(val_loader))
    history['val_morph'].append(val_morph_loss / len(val_loader))
    history['lr'].append(current_lr)
    history['grad_norm'].append(avg_grad_norm)
    
    print(f"\nEpoch {epoch+1}/{SAFE_CONFIG['num_epochs']}:")
    print(f"  Train Loss: {avg_train_loss:.6f} (Color: {train_color_loss/len(train_loader):.6f}, Morph: {train_morph_loss/len(train_loader):.6f})")
    print(f"  Val Loss:   {avg_val_loss:.6f} (Color: {val_color_loss/len(val_loader):.6f}, Morph: {val_morph_loss/len(val_loader):.6f})")
    print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
    print(f"  LR: {current_lr:.8f}")
    
    if epoch > 0:
        prev_lr = history['lr'][-2]
        if current_lr != prev_lr:
            print(f"  LR changed by scheduler: {prev_lr:.8f} -> {current_lr:.8f}")
    
    improvement_threshold = 0.001
    
    if avg_val_loss < best_val_loss * (1 - improvement_threshold):
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_epoch = epoch
        
        best_model_state = model.state_dict().copy()
        best_optimizer_state = optimizer.state_dict().copy()
        
        model_path = os.path.join(config['paths']['models_dir'], 'best_safe_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'color_normalizer_mean': color_normalizer.mean,
            'color_normalizer_std': color_normalizer.std,
            'morph_normalizer_mean': morph_normalizer.mean,
            'morph_normalizer_std': morph_normalizer.std,
            'history': history
        }, model_path)
        
        improvement_pct = (1 - avg_val_loss/best_val_loss) * 100
        print(f"  Improvement: {improvement_pct:.3f}%. Model saved.")
    else:
        patience_counter += 1
        
        if patience_counter >= 3 and current_lr > SAFE_CONFIG['min_lr'] * 2:
            new_lr = current_lr * 0.7
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"  Reducing LR manually: {new_lr:.8f}")
            patience_counter = 0
        
        print(f"  Early stopping counter: {patience_counter}/{SAFE_CONFIG['patience']}")
        
        if patience_counter >= SAFE_CONFIG['patience']:
            print(f"\nEARLY STOPPING TRIGGERED")
            print(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("  Best model state restored.")
            break
    
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(config['paths']['models_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'history': history
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    print("\nTRAINING COMPLETE")
    
    final_model_path = os.path.join(config['paths']['models_dir'], 'final_safe_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(history['train_color'], label='Color', color='green')
    axes[0, 1].plot(history['train_morph'], label='Morph', color='orange')
    axes[0, 1].set_title('Train Components')
    axes[0, 1].legend()
    
    axes[0, 2].plot(history['val_color'], label='Color', color='green', linestyle='--')
    axes[0, 2].plot(history['val_morph'], label='Morph', color='orange', linestyle='--')
    axes[0, 2].set_title('Val Components')
    axes[0, 2].legend()
    
    axes[1, 0].plot(history['lr'], color='purple')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].plot(history['grad_norm'], color='brown')
    axes[1, 1].axhline(y=SAFE_CONFIG['max_grad_norm'], color='r', linestyle='--')
    axes[1, 1].set_title('Gradient Norm')
    
    train_val_gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
    axes[1, 2].plot(train_val_gap, color='black')
    axes[1, 2].axhline(y=0, color='r', linestyle='--')
    axes[1, 2].set_title('Train-Val Gap')
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(config['paths']['output_dir'], 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        for i, (img, color_target, morph_target, filename) in enumerate(val_dataset):
            if i >= 3: break
            
            img = img.unsqueeze(0).to(device)
            color_pred, morph_pred = model(img)
            
            c_pred = color_pred.cpu().numpy()[0] * color_normalizer.std + color_normalizer.mean
            c_true = color_target.numpy() * color_normalizer.std + color_normalizer.mean
            
            print(f"\nFile: {filename}")
            color_names = ['Red', 'Green', 'Blue', 'R/G', 'R/B', 'G/B']
            for j, name in enumerate(color_names):
                error = abs(c_pred[j] - c_true[j])
                print(f"  {name}: Pred {c_pred[j]:.2f} / True {c_true[j]:.2f} (Err: {error:.2f})")
    
    history_df = pd.DataFrame(history)
    history_path = os.path.join(config['paths']['output_dir'], 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    print("\nFILES GENERATED:")
    print(f"  - {model_path}")
    print(f"  - {final_model_path}")
    print(f"  - {plot_path}")
    print(f"  - {history_path}")
    
    print(f"\nFINAL METRICS:")
    print(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
    
    if os.path.exists('temp_import.py'):
        os.remove('temp_import.py')

if __name__ == "__main__":
    main()
