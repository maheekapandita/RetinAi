import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration,TrainingArguments,Trainer
from torch.utils.data import Dataset
import json
from PIL import Image
import os

MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATASET_PATH="data/processed/idrid_train_conversations.jsonl"
OUTPUT_DIR="results/retinai-llava-v1-idrid-test"

MPS_AVAILABLE=torch.backends.mps.is_available()
DEVICE="mps" if MPS_AVAILABLE else "cpu"
print(f"Using device: {DEVICE}")

class VLMDataset(Dataset):
    def __init__(self,jsonl_path,processor):
        self.processor=processor
        self.data=[]
        full_path=os.path.join(os.path.dirname(__file__),'..',jsonl_path)
        with open(full_path,"r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        item=self.data[idx]
        image_path=os.path.join(os.path.dirname(__file__),'..',item['image'])
        image=Image.open(image_path).convert("RGB")
        prompt=f"USER: {item['conversations'][0]['value']}\nASSISTANT: {item['conversations'][1]['value']}"
        inputs=self.processor(text=prompt,images=image,return_tensors="pt",padding=True)
        inputs={k: v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"]=inputs["input_ids"].clone()
        return inputs

print(f"Loading model: {MODEL_ID}")
processor=AutoProcessor.from_pretrained(MODEL_ID)

if MPS_AVAILABLE:
    model=LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.backends.mps.is_built() else torch.float32
    )
    model=model.to("mps")
else:
    model=LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )

print("Loading and processing dataset...")
train_dataset=VLMDataset(DATASET_PATH,processor)
print(f"Dataset loaded with {len(train_dataset)} samples.")

training_args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=5,
    report_to="none",
    bf16=MPS_AVAILABLE,
    fp16=not MPS_AVAILABLE,
    push_to_hub=False,
    dataloader_pin_memory=False,
    save_steps=50,
    save_total_limit=2
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer
)

print("\n--- Starting Initial Training on M2 Pro ---")
print("This will be slow,but it's a great test to ensure everything is working.")
trainer.train()
print("\n✅ Initial training complete!")