import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer, RobertaTokenizer, ViTFeatureExtractor
from PIL import Image
from typing import List
from datasets import Dataset as HFDataset, DatasetDict

class MultimodalDataset(Dataset):
    def __init__(self, 
                 image_paths: List[str], 
                 captions: List[str], 
                 labels: List[int],
                 bert_tokenizer: BertTokenizer,
                 roberta_tokenizer: RobertaTokenizer,
                 vit_feature_extractor: ViTFeatureExtractor):
        self.image_paths = image_paths
        self.captions = captions
        self.labels = labels
        self.bert_tokenizer = bert_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.vit_feature_extractor = vit_feature_extractor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image_tensor = self.transform(image)
        caption = self.captions[idx]
        label = self.labels[idx]

        # Tokenize text
        bert_inputs = self.bert_tokenizer(caption, padding='max_length', truncation=True, return_tensors='pt')
        roberta_inputs = self.roberta_tokenizer(caption, padding='max_length', truncation=True, return_tensors='pt')

        # Process images for ViT
        vit_inputs = self.vit_feature_extractor(images=image, return_tensors='pt')

        return {
            'image1': image_tensor,
            'image2': image_tensor,
            'bert_input_ids': bert_inputs['input_ids'].squeeze(),
            'bert_attention_mask': bert_inputs['attention_mask'].squeeze(),
            'roberta_input_ids': roberta_inputs['input_ids'].squeeze(),
            'roberta_attention_mask': roberta_inputs['attention_mask'].squeeze(),
            'vit_pixel_values': vit_inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_dataset_dict(image_paths, captions, labels, bert_tokenizer, roberta_tokenizer, vit_feature_extractor, train_ratio=0.8):
    dataset = MultimodalDataset(image_paths, captions, labels, bert_tokenizer, roberta_tokenizer, vit_feature_extractor)
    
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return DatasetDict({
        'train': HFDataset.from_dict({k: [d[k].tolist() if isinstance(d[k], torch.Tensor) else d[k] for d in train_dataset] for k in train_dataset[0].keys()}),
        'validation': HFDataset.from_dict({k: [d[k].tolist() if isinstance(d[k], torch.Tensor) else d[k] for d in val_dataset] for k in val_dataset[0].keys()})
    })