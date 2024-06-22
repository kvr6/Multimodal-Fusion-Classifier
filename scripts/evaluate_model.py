import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalModel
from src.config import MultimodalConfig
from src.dataset import create_dataset_dict
from src.utils import generate_mock_data, compute_metrics
from transformers import BertTokenizer, RobertaTokenizer, ViTImageProcessor, Trainer

def get_latest_checkpoint(model_path):
    checkpoints = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(model_path, latest_checkpoint)

def evaluate_model(model_path):
    latest_checkpoint = get_latest_checkpoint(model_path)
    if latest_checkpoint is None:
        print(f"Error: No checkpoints found in {model_path}")
        return

    print(f"Using checkpoint: {latest_checkpoint}")

    # Generate mock data for evaluation
    image_paths, captions, labels = generate_mock_data(200)

    # Initialize tokenizers and feature extractor
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Create dataset
    dataset_dict = create_dataset_dict(image_paths, captions, labels, bert_tokenizer, roberta_tokenizer, vit_processor)

    # Load the model
    config = MultimodalConfig.from_pretrained(latest_checkpoint)
    model = MultimodalModel.from_pretrained(latest_checkpoint)

    # Set up trainer for evaluation
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    results = trainer.evaluate(dataset_dict['validation'])

    print("Evaluation Results:", results)
    return results

if __name__ == "__main__":
    model_path = "./results"
    evaluate_model(model_path)