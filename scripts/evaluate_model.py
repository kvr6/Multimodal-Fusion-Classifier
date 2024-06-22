from src.model import MultimodalModel
from src.config import MultimodalConfig
from src.dataset import create_dataset_dict
from src.utils import generate_mock_data, compute_metrics
from transformers import BertTokenizer, RobertaTokenizer, ViTFeatureExtractor, Trainer

def evaluate_model(model_path):
    # Generate mock data for evaluation
    image_paths, captions, labels = generate_mock_data(200)

    # Initialize tokenizers and feature extractor
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Create dataset
    dataset_dict = create_dataset_dict(image_paths, captions, labels, bert_tokenizer, roberta_tokenizer, vit_feature_extractor)

    # Load the model
    config = MultimodalConfig.from_pretrained(model_path)
    model = MultimodalModel.from_pretrained(model_path)

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
    model_path = "./your_model_directory"  # Update this to your actual model path
    evaluate_model(model_path)