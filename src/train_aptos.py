import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer, BitsAndBytesConfig
from torch.utils.data import Dataset
from torchvision import transforms
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from PIL import Image
import os

MODEL_ID="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"
DATASET_PATH="/home/saririans/RetinAI/data/processed/aptos_train_conversations.jsonl"
OUTPUT_DIR="/blue/bme6938/saririans/RetinAI_results/retinai-llava-med-v1-aptos"

class VLMDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor=processor
        self.data=[json.loads(line) for line in open(jsonl_path)]
        self.augment=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item=self.data[idx]
        image=Image.open(item['image']).convert("RGB")
        image=self.augment(image)
        prompt=item['conversations'][0]['value'] + "\n" + item['conversations'][1]['value']
        
        inputs=self.processor(text=prompt, images=image, return_tensors="pt")
        inputs={k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"]=inputs["input_ids"].clone()
        return inputs

print(f"Loading idrid model from: {MODEL_ID}")

quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16)

processor=AutoProcessor.from_pretrained(MODEL_ID)
model=LlavaForConditionalGeneration.from_pretrained(MODEL_ID,
                                                    quantization_config=quantization_config,
                                                    device_map="auto")

model=prepare_model_for_kbit_training(model)

peft_config=LoraConfig(r=64,
                       lora_alpha=128,
                       target_modules=["q_proj", 
                       "k_proj", 
                       "v_proj", 
                       "o_proj", 
                       "gate_proj", 
                       "up_proj", 
                       "down_proj"], 
                       lora_dropout=0.05,
                       bias="none",
                       task_type="CAUSAL_LM")

model=get_peft_model(model, peft_config)
model.print_trainable_parameters() 

print("Loading APTOS dataset...")
train_dataset = VLMDataset(DATASET_PATH, processor)
print(f"Dataset loaded with {len(train_dataset)} samples.")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=6,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    fp16=True,
    push_to_hub=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor.tokenizer
)

print("Starting APTOS train")
trainer.train()

print("Saving Final Model...")
trainer.save_model()
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")
