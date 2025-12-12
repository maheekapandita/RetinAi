import pandas as pd
import json
import os

BASE_DATA_DIR="/blue/bme6938/saririans/RetinAI_data"
CSV_PATH=os.path.join(BASE_DATA_DIR,"Aptos/train.csv")
IMAGE_DIR=os.path.join(BASE_DATA_DIR,"Aptos/train_images")
OUTPUT_DIR="/home/saririans/RetinAI/data/processed"
OUTPUT_FILE=os.path.join(OUTPUT_DIR,"aptos_train_conversations.jsonl")

grade_mapping={0:"no diabetic retinopathy",
               1:"mild diabetic retinopathy",
               2:"moderate diabetic retinopathy",
               3:"severe diabetic retinopathy",
               4:"proliferative diabetic retinopathy"}

print(f"Reading CSV from: {CSV_PATH}")

try:
    df=pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records.")
except FileNotFoundError:
    print(f"Error: CSV not found {CSV_PATH}")
    exit()

os.makedirs(OUTPUT_DIR,exist_ok=True)
written_count=0

print("Processing images...")
with open(OUTPUT_FILE,'w') as f:
    for index,row in df.iterrows():
        image_id=str(row['id_code']).strip()
        grade=int(row['diagnosis'])
        image_path=os.path.join(IMAGE_DIR,f"{image_id}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        grade_text=grade_mapping.get(grade,"unknown grade")
        conversation_data={"id":image_id,"image":image_path,"conversations":[{"from":"human","value":"<image>\nDescribe this retinal image."},{"from":"gpt","value":f"This is a retinal image with a diagnosis of {grade_text}."}]}
        f.write(json.dumps(conversation_data)+"\n")
        written_count+=1

print(f"APTOS Prep complete {written_count} records written to:")
print(f"   {OUTPUT_FILE}")