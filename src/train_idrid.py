import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration,TrainingArguments,Trainer
from torch.utils.data import Dataset
import json
from PIL import Image
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]="false"

BASE_DIR="/home/saririans/RetinAI"
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATASET_PATH=os.path.join(BASE_DIR,"data/processed/idrid_train_conversations.jsonl")
OUTPUT_DIR="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"

print("Training idrid:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"DATASET_PATH: {DATASET_PATH}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"Dataset exists: {os.path.exists(DATASET_PATH)}")
print("Checking CUDA")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("WARNING: CUDA not available")

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR,exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

class VLMDataset(Dataset):
    def __init__(self,jsonl_path,processor):
        self.processor=processor
        self.data=[]
        self.image_base_dir="/blue/bme6938/saririans/RetinAI_data/B. Disease Grading/1. Original Images/a. Training Set"
        print(f"Using image base directory: {self.image_base_dir}")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
        with open(jsonl_path,"r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} samples from dataset")
        if len(self.data)>0:
            print("\nFirst sample:")
            print(f"  ID: {self.data[0]['id']}")
            print(f"  Image: {self.data[0]['image']}")
            print(f"  Question: {self.data[0]['conversations'][0]['value']}")
            print(f"  Answer: {self.data[0]['conversations'][1]['value']}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        item=self.data[idx]
        image_filename=os.path.basename(item['image'])
        image_path=os.path.join(self.image_base_dir,image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image=Image.open(image_path).convert("RGB")
        conversation=item['conversations']
        human_message=conversation[0]['value']
        assistant_message=conversation[1]['value']
        prompt=f"USER: <image>\n{human_message}\nASSISTANT: {assistant_message}"
        inputs=self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=False)
        
        inputs={k:v.squeeze(0) if v.dim()>1 else v for k,v in inputs.items()}
        prompt_only=f"USER: <image>\n{human_message}\nASSISTANT:"
        prompt_inputs=self.processor(
            text=prompt_only,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=False)
        
        prompt_length=prompt_inputs['input_ids'].shape[1]
        labels=inputs["input_ids"].clone()
        labels[:prompt_length]=-100
        inputs["labels"]=labels
        return inputs

try:
    print(f"Loading model: {MODEL_ID}")
    processor=AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side="right"
    print("Processor loaded successfully")
    if torch.cuda.is_available():
        gpu_count=torch.cuda.device_count()
        print(f"Loading model on {gpu_count} GPU(s)...")
        if gpu_count>=2:
            max_memory={i:"18GiB" for i in range(gpu_count)}
            print(f"Using {gpu_count} GPUs with memory limits: {max_memory}")
        else:
            max_memory={0:"18GiB"}
        gpu_name=torch.cuda.get_device_name(0).lower()
        if "l4" in gpu_name or "a100" in gpu_name or "h100" in gpu_name:
            model_dtype=torch.bfloat16
            print("Using bfloat16 precision (GPU supports it)")
        else:
            model_dtype=torch.float16
            print("Using float16 precision")
        model=LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory)
    else:
        print("Loading model on CPU...")
        model=LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True)
    print("Model loaded successfully")
except Exception as e:
    print(f"ERROR loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    print("Loading and processing dataset...")
    train_dataset=VLMDataset(DATASET_PATH,processor)
    print(f"Dataset loaded with {len(train_dataset)} samples.")
    if len(train_dataset)==0:
        raise ValueError("Dataset is empty! Check your data file.")
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    import traceback
    traceback.print_exc()
    raise

if torch.cuda.is_available():
    gpu_name=torch.cuda.get_device_name(0).lower()
    use_bf16="l4" in gpu_name or "a100" in gpu_name or "h100" in gpu_name
    print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")
else:
    use_bf16=False

try:
    import bitsandbytes as bnb
    optimizer_8bit=True
    print("Using 8-bit optimizer (bitsandbytes available)")
except ImportError:
    optimizer_8bit=False
    print("8-bit optimizer not available, using standard optimizer")

training_args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    fp16=torch.cuda.is_available() and not use_bf16,
    bf16=torch.cuda.is_available() and use_bf16,
    push_to_hub=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    dataloader_drop_last=True,
    dataloader_pin_memory=False,
    optim="adamw_8bit" if optimizer_8bit else "adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    logging_first_step=True,
    load_best_model_at_end=False)

class VLMDataCollator:
    def __init__(self,processor):
        self.processor=processor
        self.pad_token_id=processor.tokenizer.pad_token_id
    def __call__(self,features):
        max_length=max(f["input_ids"].shape[0] for f in features)
        batch={}
        input_ids_list=[]
        attention_mask_list=[]
        labels_list=[]
        for f in features:
            input_ids=f["input_ids"]
            attention_mask=f["attention_mask"]
            labels=f["labels"]
            pad_length=max_length-input_ids.shape[0]
            if pad_length>0:
                input_ids=torch.cat([
                    input_ids,
                    torch.full((pad_length,),self.pad_token_id,dtype=input_ids.dtype)
                ])
                attention_mask=torch.cat([
                    attention_mask,
                    torch.zeros(pad_length,dtype=attention_mask.dtype)
                ])
                labels=torch.cat([
                    labels,
                    torch.full((pad_length,),-100,dtype=labels.dtype)
                ])
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        batch["input_ids"]=torch.stack(input_ids_list)
        batch["attention_mask"]=torch.stack(attention_mask_list)
        batch["labels"]=torch.stack(labels_list)
        batch["pixel_values"]=torch.stack([f["pixel_values"] for f in features])
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"]=torch.stack([f["image_grid_thw"] for f in features])
        return batch

data_collator=VLMDataCollator(processor)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator)

print("train start")

try:
    trainer.train()
    print("="*50)
    print("Training completed successfully!")
    print("="*50)
except Exception as e:
    print(f"ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    print("Saving model...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")
except Exception as e:
    print(f"ERROR saving model: {e}")
    import traceback
    traceback.print_exc()
    raise