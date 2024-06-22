import os
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generate_mock_data(num_samples: int = 1000, output_dir: str = 'data/mock_images'):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = [f"{output_dir}/mock_image_{i}.jpg" for i in range(num_samples)]
    captions = [f"This is a mock caption for image {i}" for i in range(num_samples)]
    labels = np.random.randint(0, 5, num_samples).tolist()
    
    # Create mock images
    for path in image_paths:
        img = Image.new('RGB', (224, 224), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        img.save(path)
    
    return image_paths, captions, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }