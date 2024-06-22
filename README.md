# Multimodal Fusion Classifier

A PyTorch-based implementation of a multimodal classifier that combines vision and language models for advanced classification tasks. This project integrates pre-trained ResNet, BERT, RoBERTa, and ViT models to process both image and text inputs, fusing their features for final classification.

## Project Structure

```
multimodal-fusion-classifier/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
├── scripts/
│   ├── train_model.py
│   └── evaluate_model.py
├── data/
│   └── mock_images/
├── results/
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multimodal-fusion-classifier.git
   cd multimodal-fusion-classifier
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:

```
python scripts/train_model.py
```

This script generates mock data, creates the dataset, trains the model, and saves checkpoints in the `results/` directory.

### Evaluation

To evaluate the trained model:

```
python scripts/evaluate_model.py
```

This script loads the latest checkpoint from the `results/` directory and evaluates the model's performance on a mock evaluation dataset.

## Model Architecture

The multimodal classifier combines the following pre-trained models:

- Two ResNet50 models for image processing
- BERT for text processing
- RoBERTa for additional text processing
- Vision Transformer (ViT) for image processing

The outputs of these models are concatenated and passed through a multi-layer perceptron (MLP) for final classification.

## Data

The current implementation uses mock data for demonstration purposes. For real-world usage, replace the `generate_mock_data` function in `src/utils.py` with your actual data loading process.

## Performance Metrics

The model's performance is evaluated using the following metrics:

- Accuracy
- F1 Score
- Precision
- Recall

These metrics are computed using the `compute_metrics` function in `src/utils.py`.

## Future Improvements

- Replace mock data with real dataset
- Experiment with different model architectures and hyperparameters
- Implement data augmentation techniques for image data
- Fine-tune pre-trained models instead of freezing them
- Optimize for specific downstream tasks
- Upload the trained model to Hugging Face Hub for easier sharing and deployment
- Implement more robust error handling and logging
- Add unit tests and integration tests
- Create a demo application or API endpoint for easy model inference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
