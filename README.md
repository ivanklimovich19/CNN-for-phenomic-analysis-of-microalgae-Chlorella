# Chlorella Phenotype Analyzer

Deep learning system for automatic phenotype analysis of Chlorella microalgae from microscopic images.

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/chlorella-analyzer.git
cd chlorella-analyzer
pip install -r requirements.txt
```

### Configuration
Edit `config.yaml`:
```yaml
paths:
  images_dir: ./lerset/      # Input images
  output_dir: ./output/       # Results
  models_dir: ./models/       # Saved models
```

### Train Model
```bash
python learn_model.py
```

### Analyze Images
Interactive mode:
```bash
python use_model.py
```

Or in code:
```python
from use_model import ChlorellaPhenotypePredictor

predictor = ChlorellaPhenotypePredictor()
result = predictor.predict('image.jpg')
print(result['color_features'])  # RGB values
print(result['morph_features'])   # Cell count, areas, etc.

# Batch analysis
df = predictor.analyze_directory('./lerset/', 'results.csv')
```

## Output Features

**Color** (6): Red, Green, Blue, R/G, R/B, G/B  
**Morphological** (7): Cell Count, Mean/Median/Min/Max/Std/Total Area

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV, NumPy, Pandas
