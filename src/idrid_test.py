import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration
from PIL import Image
import os
import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

MODEL_DIR="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"
TEST_IMAGE_DIR="/blue/bme6938/saririans/RetinAI_data/B. Disease Grading/1. Original Images/b. Testing Set"
TEST_CSV_PATH="/blue/bme6938/saririans/RetinAI_data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

print("IDRiD Diabetic retinopathy testing")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("CUDA not available")
print(f"Loading model from: {MODEL_DIR}")
try:
    processor=AutoProcessor.from_pretrained(MODEL_DIR)
    if torch.cuda.is_available():
        model=LlavaForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto")
    else:
        model=LlavaForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True).to(DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit()

def parse_grade_from_text(text):
    text=text.lower()
    if "proliferative" in text:
        return 4
    if "severe" in text:
        return 3
    if "moderate" in text:
        return 2
    if "mild" in text:
        return 1
    if "no diabetic" in text or "no dr" in text or "grade 0" in text:
        return 0
    return -1

grade_mapping={
    0:"No DR",
    1:"Mild DR",
    2:"Moderate DR",
    3:"Severe DR",
    4:"Proliferative DR"}

print(f"Loading test set: {TEST_CSV_PATH}")
try:
    df=pd.read_csv(TEST_CSV_PATH)
    print(f"Found {len(df)} test samples")
except FileNotFoundError:
    print(f"ERROR: Could not find CSV at '{TEST_CSV_PATH}'")
    exit()

y_true=[]
y_pred=[]
failed_samples=[]

print(f"Starting evaluation on {len(df)} test images...")
print("="*60)

for index,row in tqdm(df.iterrows(),total=df.shape[0],desc="Evaluating"):
    try:
        image_name=row['Image name'].strip()+".jpg"
        true_grade=int(row['Retinopathy grade'])
        image_path=os.path.join(TEST_IMAGE_DIR,image_name)
        if not os.path.exists(image_path):
            print(f"\nWarning: Skipping missing image {image_name}")
            y_true.append(true_grade)
            y_pred.append(-1)
            failed_samples.append((image_name,"Image not found"))
            continue
        image=Image.open(image_path).convert("RGB")
        prompt="USER: <image>\nDescribe this retinal image.\nASSISTANT:"
        inputs=processor(text=prompt,images=image,return_tensors="pt")
        if DEVICE=="cuda":
            inputs={k:v.to(DEVICE) for k,v in inputs.items()}
        with torch.no_grad():
            output=model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0)

        response_text=processor.decode(output[0],skip_special_tokens=True)
        if "ASSISTANT:" in response_text:
            response_text=response_text.split("ASSISTANT:")[-1].strip()
        predicted_grade=parse_grade_from_text(response_text)
        y_true.append(true_grade)
        y_pred.append(predicted_grade)
        if predicted_grade==-1:
            failed_samples.append((image_name,response_text[:100]))
    except Exception as e:
        print(f"\nError processing {image_name}: {e}")
        y_true.append(true_grade)
        y_pred.append(-1)
        failed_samples.append((image_name,f"Exception: {str(e)}"))

print("Evaluation Complete")


valid_predictions=[(t,p) for t,p in zip(y_true,y_pred) if p!=-1]
if valid_predictions:
    valid_true,valid_pred=zip(*valid_predictions)
    valid_accuracy=accuracy_score(valid_true,valid_pred)
    print(f"Accuracy (valid predictions only): {valid_accuracy*100:.2f}%")
    print(f"Valid predictions: {len(valid_predictions)}/{len(y_true)}")
else:
    print("No valid predictions")

overall_accuracy=accuracy_score(y_true,y_pred)
print(f"Overall accuracy: {overall_accuracy*100:.2f}%")

print("\n"+"="*60)
print("Confusion Matrix:")
print("="*60)
cm=confusion_matrix(y_true,y_pred,labels=[0,1,2,3,4,-1])
print("Labels:          0    1    2    3    4  (Err)")
print(cm)
print("\n(Rows=true labls, Columns=pred labels)")
print("(-1 = Model output not parsed)")

print("\n"+"="*60)
print("Classification Report:")
print("="*60)
report=classification_report(
    y_true,
    y_pred,
    labels=[0,1,2,3,4],
    target_names=list(grade_mapping.values()),
    zero_division=0)
print(report)

if failed_samples:
    print("\n"+"="*60)
    print(f"Failed to parse {len(failed_samples)} samples:")
    print("="*60)
    for img_name,reason in failed_samples[:10]:
        print(f"  {img_name}: {reason}")
    if len(failed_samples)>10:
        print(f"  ... and {len(failed_samples)-10} more")

print("testing done")

