import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration,BitsAndBytesConfig
from peft import PeftModel
import json
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

BASE_MODEL_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"
ADAPTER_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-aptos"
DATA_JSONL="/home/saririans/RetinAI/data/processed/aptos_train_conversations.jsonl"

text_to_grade={"no diabetic retinopathy":0,"mild diabetic retinopathy":1,"moderate diabetic retinopathy":2,"severe diabetic retinopathy":3,"proliferative diabetic retinopathy":4}

print("Loading Model...")
quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
processor=AutoProcessor.from_pretrained(BASE_MODEL_PATH)
model=LlavaForConditionalGeneration.from_pretrained(BASE_MODEL_PATH,quantization_config=quantization_config,device_map="auto")
model=PeftModel.from_pretrained(model,ADAPTER_PATH)
model.eval()

print(f"Reading data from: {DATA_JSONL}")
all_data=[json.loads(line) for line in open(DATA_JSONL)]

y_true=[]
y_pred=[]
valid_count=0

print(f"Starting inference on {len(all_data)} images...")

for i,item in enumerate(tqdm(all_data)):
    gt_text=item['conversations'][1]['value'].lower()
    true_label=-1
    for key,val in text_to_grade.items():
        if key in gt_text:
            true_label=val
            break
    if true_label==-1: continue

    image_path=item['image']
    prompt="<image> Describe this retinal image."

    try:
        image=Image.open(image_path).convert("RGB")
        inputs=processor(text=prompt,images=image,return_tensors="pt").to("cuda")
        with torch.no_grad():
            generate_ids=model.generate(**inputs,max_new_tokens=64)
        generated_ids=generate_ids[:,inputs.input_ids.shape[1]:]
        response=processor.batch_decode(generated_ids,skip_special_tokens=True)[0].lower().strip()
        
        pred_label=-1
        for class_name,label_idx in text_to_grade.items():
            if class_name in response:
                pred_label=label_idx
                break
        if pred_label==-1:
            if "no diabetic" in response: pred_label=0
            elif "mild" in response: pred_label=1
            elif "moderate" in response: pred_label=2
            elif "severe" in response: pred_label=3
            elif "proliferative" in response: pred_label=4

        if i<5:
            print(f"[DEBUG {i}]")
            print(f"GT Text: '{gt_text}' -> Label: {true_label}")
            print(f"Model Output: '{response}'")
            print(f"Parsed Prediction: {pred_label}")

        if pred_label!=-1:
            valid_count+=1
            y_pred.append(pred_label)
            y_true.append(true_label)
    except Exception as e:
        print(f"Error: {e}")

if valid_count>0:
    print(f"\nValid Predictions: {valid_count}/{len(all_data)}")
    print(f"Accuracy: {accuracy_score(y_true,y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true,y_pred,target_names=list(text_to_grade.keys())))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true,y_pred))
else:
    print(" No valid predictions found. Check parsing logic.")