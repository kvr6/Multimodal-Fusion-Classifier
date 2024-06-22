import torch
import torch.nn as nn
from transformers import PreTrainedModel, BertModel, RobertaModel, ViTModel
from torchvision.models import resnet50, ResNet50_Weights
from .config import MultimodalConfig

class MultimodalModel(PreTrainedModel):
    config_class = MultimodalConfig
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        
        # Vision models
        self.resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])
        self.resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])
        
        # Text models
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.roberta = RobertaModel.from_pretrained(config.roberta_model_name)
        
        # Vision Transformer
        self.vit = ViTModel.from_pretrained(config.vit_model_name)
        
        # Freeze all base models
        for model in [self.resnet1, self.resnet2, self.bert, self.roberta, self.vit]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Calculate total input size for the MLP
        self.total_feature_size = (
            2048 +  # ResNet output size
            2048 +  # Second ResNet output size
            config.hidden_size +  # BERT hidden size
            config.hidden_size +  # RoBERTa hidden size
            config.hidden_size    # ViT hidden size
        )
        
        # MLP layers (only this part will be fine-tuned)
        self.mlp = nn.Sequential(
            nn.Linear(self.total_feature_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_labels)
        )

    def forward(self, image1, image2, bert_input_ids, bert_attention_mask, 
                roberta_input_ids, roberta_attention_mask, vit_pixel_values, labels=None):
        # Process images
        with torch.no_grad():
            resnet1_features = self.resnet1(image1).squeeze(-1).squeeze(-1)
            resnet2_features = self.resnet2(image2).squeeze(-1).squeeze(-1)
            
            # Process text
            bert_output = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask).last_hidden_state[:, 0, :]
            roberta_output = self.roberta(input_ids=roberta_input_ids, attention_mask=roberta_attention_mask).last_hidden_state[:, 0, :]
            
            # Process with ViT
            vit_output = self.vit(pixel_values=vit_pixel_values).last_hidden_state[:, 0, :]
        
        # Concatenate all features
        combined_features = torch.cat([
            resnet1_features, resnet2_features, bert_output, roberta_output, vit_output
        ], dim=1)
        
        # Pass through MLP
        logits = self.mlp(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}