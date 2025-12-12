import pandas as pd
import json
import os

base_dir=os.path.join(os.path.dirname(__file__),'..')
csv_path=os.path.join(base_dir,"data/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
image_dir=os.path.join(base_dir,"data/IDRiD/B. Disease Grading/1. Original Images/a. Training Set")
output_file=os.path.join(base_dir,"data/processed/idrid_train_conversations.jsonl")

grade_mapping={0: "no diabetic retinopathy",
               1: "mild diabetic retinopathy",
               2: "moderate diabetic retinopathy",
               3: "severe diabetic retinopathy",
               4: "proliferative diabetic retinopathy"}

try:
    df=pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from CSV")
except FileNotFoundError:
    print(f"ERROR: Could not find CSV file at '{csv_path}'")
    exit()

os.makedirs(os.path.dirname(output_file),exist_ok=True)
written_count=0
with open(output_file,'w') as f:
    for index,row in df.iterrows():
        image_id=row['Image name'].strip()
        grade=row['Retinopathy grade']
        image_filename=f"{image_id}.jpg"
        image_abs_path=os.path.join(image_dir,image_filename)
        image_rel_path=f"data/IDRiD/B. Disease Grading/1. Original Images/a. Training Set/{image_filename}"
        if not os.path.exists(image_abs_path) or pd.isna(grade):
            continue
        
        grade_text=grade_mapping.get(int(grade),"unknown grade")
        conversation_data={
            "id": image_id,
            "image": image_rel_path,
            "conversations": [
                {"from": "human", "value": "Describe this retinal image."},
                {"from": "gpt", "value": f"This is a retinal image with a diagnosis of {grade_text}."}]}
        f.write(json.dumps(conversation_data)+"\n")
        written_count+=1

print(f"{written_count} records written to '{output_file}'")
