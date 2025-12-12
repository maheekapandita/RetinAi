import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import gradio as gr

MODEL_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

print("Loading Diabetic Retinopathy Classification Model")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Loading model: {MODEL_PATH}")
processor=AutoProcessor.from_pretrained(MODEL_PATH)

if torch.cuda.is_available():
    model=LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto")
else:
    model=LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True).to(DEVICE)

model.eval()
print("Model loaded")


def parse_grade_from_text(text):
    text=text.lower()
    if "proliferative" in text:
        return "Proliferative DR (Grade 4)",4
    if "severe" in text:
        return "Severe DR (Grade 3)",3
    if "moderate" in text:
        return "Moderate DR (Grade 2)",2
    if "mild" in text:
        return "Mild DR (Grade 1)",1
    if "no diabetic" in text or "no dr" in text:
        return "No DR (Grade 0)",0
    return "Unable to classify",-1


def get_severity_color(grade):
    colors={
        0:"#28a745",
        1:"#ffc107",
        2:"#fd7e14",
        3:"#dc3545",
        4:"#6f42c1",
        -1:"#6c757d"} # from google 
    return colors.get(grade,"#6c757d")


def get_recommendation(grade):
    recommendations={
        0:"No signs of diabetic retinopathy detected. Continue regular annual eye exams.",
        1:"Mild diabetic retinopathy detected. Follow up in 6-12 months recommended.",
        2:"Moderate diabetic retinopathy detected. Follow up in 3-6 months recommended. Consider tighter diabetes control.",
        3:"Severe diabetic retinopathy detected. Immediate referral to ophthalmologist recommended. Follow up in 1-3 months.",
        4:"Proliferative diabetic retinopathy detected. Urgent referral to retinal specialist required. May need laser treatment or injections.",
        -1:"Unable to classify. "}
    return recommendations.get(grade,"Classification uncertain.")

def classify_retinal_image(image):
    if image is None:
        return "Please upload an image"
    try:
        if not isinstance(image,Image.Image):
            image=Image.fromarray(image)
        image=image.convert("RGB")
        prompt="USER: <image>\nDescribe this retinal image.\nASSISTANT:"
        inputs=processor(text=prompt,images=image,return_tensors="pt")
        
        if DEVICE=="cuda":
            inputs={k:v.to(DEVICE) for k,v in inputs.items()}
        
        with torch.no_grad():
            output=model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=1.0)
        
        response_text=processor.decode(output[0],skip_special_tokens=True)
        if "ASSISTANT:" in response_text:
            response_text=response_text.split("ASSISTANT:")[-1].strip()
        diagnosis,grade=parse_grade_from_text(response_text)
        color=get_severity_color(grade)
        recommendation=get_recommendation(grade)
        diagnosis_html=f'<div style="padding: 20px; background-color: {color}; color: white; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin: 10px 0;">{diagnosis}</div>'
        recommendation_html=f'<div style="padding: 15px; background-color: #212529; border-left: 4px solid {color}; border-radius: 5px; margin: 10px 0;"><strong>Recommendation:</strong><br>{recommendation}</div>'
        full_response_html=f'<div style="padding: 15px; background-color: #212529; border-radius: 5px; margin: 10px 0;"><strong>Model Output:</strong><br>{response_text}</div>'
        
        return diagnosis_html,recommendation_html,full_response_html
        
    except Exception as e:
        error_html=f'<div style="padding: 20px; background-color: #dc3545; color: white; border-radius: 10px;"><strong>Error:</strong> {str(e)}</div>'
        return error_html,"",""


with gr.Blocks(title="Diabetic Retinopathy Classification",theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Diabetic Retinopathy Classification System
        
        Upload a retinal fundus image to classify the severity of diabetic retinopathy.
        
        **Grades:**
        - Grade 0: No diabetic retinopathy
        - Grade 1: Mild diabetic retinopathy
        - Grade 2: Moderate diabetic retinopathy
        - Grade 3: Severe diabetic retinopathy
        - Grade 4: Proliferative diabetic retinopathy
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input=gr.Image(
                label="Upload Retinal Image",
                type="pil",
                height=400)
            classify_btn=gr.Button(
                "Classify Image",
                variant="primary",
                size="lg")
            gr.Markdown(
                """
                ### Instructions:
                1. Upload a retinal fundus image
                2. Click Classify Image
                3. Review the diagnosis and recommendations
                """)
        with gr.Column(scale=1):
            diagnosis_output=gr.HTML(label="Diagnosis")
            recommendation_output=gr.HTML(label="Clinical Recommendation")
            response_output=gr.HTML(label="Detailed Model Response")
    
    classify_btn.click(
        fn=classify_retinal_image,
        inputs=image_input,
        outputs=[diagnosis_output,recommendation_output,response_output])

if __name__=="__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True)