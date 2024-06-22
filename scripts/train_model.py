import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Trainer, TrainingArguments
from src.config import MultimodalConfig
from src.model import MultimodalModel
from src.dataset import create_dataset_dict
from src.utils import generate_mock_data, compute_metrics
from transformers import BertTokenizer, RobertaTokenizer, ViTFeatureExtractor

def train_model():
    # Generate mock data
    image_paths, captions, labels = generate_mock_data(1000)

    # Initialize tokenizers and feature extractor
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Create dataset
    dataset_dict = create_dataset_dict(image_paths, captions, labels, bert_tokenizer, roberta_tokenizer, vit_feature_extractor)

    # Initialize model
    config = MultimodalConfig(num_labels=5)
    model = MultimodalModel(config)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./your_model_directory")

    print("Training completed and model saved!")

if __name__ == "__main__":
    train_model()