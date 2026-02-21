import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import yaml
import glob
import pandas as pd
from torch.serialization import add_safe_globals
import numpy.core.multiarray
add_safe_globals([numpy.core.multiarray._reconstruct])

class ChlorellaPhenotypePredictor:
    def __init__(self, model_path='models/best_safe_model.pth', config_path='config.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = self._load_model(model_path)
        self.normalizers = self._load_normalizers(model_path)

    def _load_model(self, model_path):
        class SafeChlorellaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = models.resnet18(weights=None)
                self.backbone.fc = nn.Identity()
                self.bn = nn.BatchNorm1d(512)
                
                self.color_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 6)
                )
                
                self.morph_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 7)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                features = self.bn(features)
                color_pred = torch.tanh(self.color_head(features)) * 3
                morph_pred = torch.tanh(self.morph_head(features)) * 3
                return color_pred, morph_pred
        
        model = SafeChlorellaModel()
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                model.load_state_dict(state_dict)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}. Using random initialization.")
        else:
            print(f"Warning: Model file not found at {model_path}. Using random initialization.")
            
        model.to(self.device)
        model.eval()
        return model
    
    def _load_normalizers(self, model_path):
        normalizers = {
            'color': {'mean': np.zeros(6), 'std': np.ones(6)},
            'morph': {'mean': np.zeros(7), 'std': np.ones(7)}
        }
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'color_normalizer_mean' in checkpoint:
                    normalizers['color'] = {
                        'mean': checkpoint['color_normalizer_mean'],
                        'std': checkpoint['color_normalizer_std']
                    }
                    normalizers['morph'] = {
                        'mean': checkpoint['morph_normalizer_mean'],
                        'std': checkpoint['morph_normalizer_std']
                    }
                    print("Normalizers loaded from checkpoint.")
            except Exception as e:
                print(f"Error loading normalizers: {e}. Using default values.")
        
        return normalizers
    
    def preprocess_image(self, image_path):
   
        try:
            if isinstance(image_path, str):
                img = Image.open(image_path).convert('RGB')
            else:
                img = image_path.convert('RGB')
            
            img_tensor = self.transform(img)
            
            return img_tensor
            
        except Exception as e:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            return self.transform(img)
    
    def predict(self, image_path, return_normalized=False):
        
        img_tensor = self.preprocess_image(image_path)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Добавляем batch dimension
        
        with torch.no_grad():
            color_pred_norm, morph_pred_norm = self.model(img_tensor)
        
        color_pred_norm = color_pred_norm.cpu().numpy()[0]
        morph_pred_norm = morph_pred_norm.cpu().numpy()[0]
        
        if return_normalized:
            return {
                'color_features_norm': color_pred_norm,
                'morph_features_norm': morph_pred_norm
            }
        
        color_pred = self._denormalize(color_pred_norm, 'color')
        morph_pred = self._denormalize(morph_pred_norm, 'morph')
        
        result = {
            'filename': os.path.basename(image_path) if isinstance(image_path, str) else 'image',
            'color_features': {
                'Red': float(color_pred[0]),
                'Green': float(color_pred[1]),
                'Blue': float(color_pred[2]),
                'R/G': float(color_pred[3]),
                'R/B': float(color_pred[4]),
                'G/B': float(color_pred[5])
                    },
            'morph_features': {
                'Cell_Count': int(round(float(morph_pred[0]))),
                'Mean_Area': float(morph_pred[1]),
                'Median_Area': float(morph_pred[2]),
                'Min_Area': float(morph_pred[3]),
                'Max_Area': float(morph_pred[4]),
                'Std_Area': float(morph_pred[5]),
                'Total_Area': float(morph_pred[6])
            },
            'predictions_norm': {
                'color': color_pred_norm.tolist(),
                'morph': morph_pred_norm.tolist()
            }
        }
        
        return result
    
    def predict_batch(self, image_paths, batch_size=8):
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            print(f"Processing images {i+1}-{min(i+batch_size, len(image_paths))}/{len(image_paths)}")
            
            for img_path in batch_paths:
                try:
                    img_tensor = self.preprocess_image(img_path)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                    batch_images.append(self.transform(img))
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                color_preds_norm, morph_preds_norm = self.model(batch_tensor)
            
            for j, (color_norm, morph_norm) in enumerate(zip(color_preds_norm, morph_preds_norm)):
                color_pred = self._denormalize(color_norm.cpu().numpy(), 'color')
                morph_pred = self._denormalize(morph_norm.cpu().numpy(), 'morph')
                
                result = {
                    'filename': os.path.basename(batch_paths[j]),
                    'color_features': {
                        'Red': float(color_pred[0]),
                        'Green': float(color_pred[1]),
                        'Blue': float(color_pred[2]),
                        'R/G': float(color_pred[3]),
                        'R/B': float(color_pred[4]),
                        'G/B': float(color_pred[5])
                    },
                    'morph_features': {
                        'Cell_Count': int(round(float(morph_pred[0]))),
                        'Mean_Area': float(morph_pred[1]),
                        'Median_Area': float(morph_pred[2]),
                        'Min_Area': float(morph_pred[3]),
                        'Max_Area': float(morph_pred[4]),
                        'Std_Area': float(morph_pred[5]),
                        'Total_Area': float(morph_pred[6])
                    }
                }
                results.append(result)
        
        return results
    
    def _denormalize(self, normalized_data, feature_type):
        if feature_type in self.normalizers:
            mean = self.normalizers[feature_type]['mean']
            std = self.normalizers[feature_type]['std']
            return normalized_data * std + mean
        return normalized_data
    
    def save_predictions(self, predictions, output_path='predictions.csv'):
        rows = []
        for pred in predictions:
            row = {'filename': pred['filename']}
            for key, value in pred['color_features'].items():
                row[f'pred_{key}'] = value
            for key, value in pred['morph_features'].items():
                row[f'pred_{key}'] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        return df
    
    def analyze_directory(self, directory_path, output_csv='all_predictions.csv'):
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return None
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, ext)))
            image_paths.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        if not image_paths:
            print("No images found.")
            return None
        
        all_predictions = self.predict_batch(image_paths)
        df = self.save_predictions(all_predictions, output_csv)
        self._print_statistics(all_predictions)
        
        return df
    
    def _print_statistics(self, predictions):
        if not predictions:
            return
        
        print("\nPREDICTION STATISTICS:")
        
        color_stats = {}
        for pred in predictions:
            for key, value in pred['color_features'].items():
                if key not in color_stats:
                    color_stats[key] = []
                color_stats[key].append(value)
        
        print("Color Features (mean ± std):")
        for key, values in color_stats.items():
            print(f"  {key}: {np.mean(values):.2f} ± {np.std(values):.2f}")
        
        morph_stats = {}
        for pred in predictions:
            for key, value in pred['morph_features'].items():
                if key not in morph_stats:
                    morph_stats[key] = []
                morph_stats[key].append(value)
        
        print("\nMorphological Features (mean ± std):")
        for key, values in morph_stats.items():
            print(f"  {key}: {np.mean(values):.2f} ± {np.std(values):.2f}")
        
        print(f"\nTotal images analyzed: {len(predictions)}")


def main():

      try:
        predictor = ChlorellaPhenotypePredictor()
      except Exception as e:
        print(f"Error initializing predictor: {e}")
        return
    
      while True:
        print("\nMAIN MENU:")
        print("1. Directory analysis")
        print("2. Single image analysis")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            folder_path = input("Enter directory path: ").strip().strip('\'"')
            if not folder_path or not os.path.exists(folder_path):
                print(f"Invalid path: {folder_path}")
                continue
            
            output_file = input("Enter output CSV name (default: all_predictions.csv): ").strip()
            if not output_file:
                output_file = 'all_predictions.csv'
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            
            results = predictor.analyze_directory(folder_path, output_file)
            if results is not None:
                print(f"Analysis complete. Results saved to: {output_file}")
                print(results.head())
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip().strip('\'"')
            if not image_path or not os.path.exists(image_path):
                print(f"Invalid path: {image_path}")
                continue
            
            result = predictor.predict(image_path)
            
            print("\nANALYSIS RESULTS:")
            print("Color Features:")
            for key, value in result['color_features'].items():
                print(f"  {key}: {value:.3f}")
            
            print("Morphological Features:")
            for key, value in result['morph_features'].items():
                val_str = f"{value}" if key == 'Cell_Count' else f"{value:.2f}"
                print(f"  {key}: {val_str}")
            
            save_choice = input("\nSave to CSV? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                output_file = input("Enter file name (default: single_prediction.csv): ").strip()
                if not output_file:
                    output_file = 'single_prediction.csv'
                if not output_file.endswith('.csv'):
                    output_file += '.csv'
                
                df = pd.DataFrame([{
                    'filename': result['filename'],
                    **{f'color_{k}': v for k, v in result['color_features'].items()},
                    **{f'morph_{k}': v for k, v in result['morph_features'].items()}
                }])
                df.to_csv(output_file, index=False)
                print(f"Result saved to: {output_file}")
        
        elif choice == '3':
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()