import pandas as pd
import json
import os

BASE_DATA_DIR="/blue/bme6938/saririans/RetinAI_data"
CSV_PATH=os.path.join(BASE_DATA_DIR,"Aptos/test.csv")
IMAGE_DIR=os.path.join(BASE_DATA_DIR,"Aptos/test_images")
OUTPUT_DIR="/home/saririans/RetinAI/data/processed"
OUTPUT_FILE=os.path.join(OUTPUT_DIR,"aptos_test_conversations.jsonl")

print(f"Reading CSV from: {CSV_PATH}")

try:
    df=pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records.")
except FileNotFoundError:
    print(f"Error: CSV not found at {CSV_PATH}")
    exit()

os.makedirs(OUTPUT_DIR,exist_ok=True)

written_count=0
print("Processing images...")

with open(OUTPUT_FILE,'w') as f:
    for index,row in df.iterrows():
        image_id=str(row['id_code']).strip()
        image_path=os.path.join(IMAGE_DIR,f"{image_id}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        conversation_data={
            "id":image_id,
            "image":image_path,
            "conversations":[
                {"from":"human",
                 "value":"<image>\nDescribe this retinal image."}]}
        f.write(json.dumps(conversation_data)+"\n")
        written_count+=1

print(f"APTOS Test Prep complete {written_count} records written to:")
print(f"   {OUTPUT_FILE}")