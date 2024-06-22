from transformers import PretrainedConfig

class MultimodalConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(
        self,
        vision_model_name: str = "microsoft/resnet-50",
        bert_model_name: str = "bert-base-uncased",
        roberta_model_name: str = "roberta-base",
        vit_model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 5,
        hidden_size: int = 768,
        num_hidden_layers: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.bert_model_name = bert_model_name
        self.roberta_model_name = roberta_model_name
        self.vit_model_name = vit_model_name
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers