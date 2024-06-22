from transformers import AutoModel, AutoConfig
from huggingface_hub import HfApi

def upload_to_hub(model_path, hub_model_id):
    print(f"Uploading model to Hugging Face Hub as {hub_model_id}...")
    
    # Load the model and config
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    # Push to hub
    model.push_to_hub(hub_model_id)
    config.push_to_hub(hub_model_id)
    
    print(f"Model successfully uploaded to {hub_model_id}")

if __name__ == "__main__":
    model_path = "./your_model_directory"
    hub_model_id = "your-username/multimodal-classifier"
    upload_to_hub(model_path, hub_model_id)