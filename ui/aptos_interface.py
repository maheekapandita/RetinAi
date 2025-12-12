import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration,BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import gradio as gr
import os
from threading import Thread

BASE_MODEL_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-v1-idrid-test"
ADAPTER_PATH="/blue/bme6938/saririans/RetinAI_results/retinai-llava-med-v1-aptos"
LOGO_PATH="Gemini_Generated_Image_qz1l3nqz1l3nqz1l.jpg"

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None
model_loading = False
model_loaded = False

def load_model():
    global processor, model, model_loading, model_loaded
    
    if model_loaded or model_loading:
        return
    
    model_loading = True
    print("\n" + "="*50)
    print("Loading RetinAI System")
    print(f"Device: {DEVICE}")
    print("="*50)
    
    try:
        print("\n[1/3] Loading Base Model...")
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
        
        processor=AutoProcessor.from_pretrained(BASE_MODEL_PATH)
        model_temp=LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("✓ Base model loaded")
        
        print("\n[2/3] Loading APTOS Adapter...")
        model_temp=PeftModel.from_pretrained(model_temp,ADAPTER_PATH)
        model_temp.eval()
        print("✓ Adapter loaded")
        
        globals()['processor'] = processor
        globals()['model'] = model_temp
        
        model_loaded = True
        model_loading = False
        print("\n[3/3] ✓ Model Successfully Loaded and Ready!")
        print("="*50 + "\n")
        
    except Exception as e:
        model_loading = False
        print(f"\n✗ Error loading model: {str(e)}")
        raise

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
    colors={0:"#00E676",1:"#FFEA00",2:"#FF9100",3:"#FF3D00",4:"#D500F9",-1:"#9E9E9E"}
    return colors.get(grade,"#9E9E9E")

def get_recommendation(grade):
    recommendations={
        0:"No signs of diabetic retinopathy detected. Continue regular annual eye exams.",
        1:"Mild diabetic retinopathy detected. Follow up in 6-12 months recommended.",
        2:"Moderate diabetic retinopathy detected. Follow up in 3-6 months recommended. Consider tighter diabetes control.",
        3:"Severe diabetic retinopathy detected. Immediate referral to ophthalmologist recommended. Follow up in 1-3 months.",
        4:"Proliferative diabetic retinopathy detected. Urgent referral to retinal specialist required. May need laser treatment or injections.",
        -1:"Unable to classify. Please try another image."
    }
    return recommendations.get(grade,"Classification uncertain.")

def classify_retinal_image(image):
    global model, processor, model_loaded, model_loading
    
    if image is None:
        return "Please upload an image","",""
    
    if model_loading:
        loading_html='<div style="padding: 20px; background-color: #FF9800; color: white; border-radius: 10px;"><strong>⏳ Model is still loading...</strong><br>Please wait a moment and try again.</div>'
        return loading_html,"",""
    
    if not model_loaded:
        loading_html='<div style="padding: 20px; background-color: #2196F3; color: white; border-radius: 10px;"><strong>🔄 Loading model...</strong><br>This will take 2-5 minutes on first run.</div>'
        Thread(target=load_model, daemon=True).start()
        return loading_html,"",""
    
    try:
        if not isinstance(image,Image.Image):
            image=Image.fromarray(image)
        image=image.convert("RGB")
        
        prompt="<image>\nDescribe this retinal image.\nThis is"
        
        inputs=processor(text=prompt,images=image,return_tensors="pt")
        if DEVICE=="cuda":
            inputs={k:v.to(DEVICE) for k,v in inputs.items()}
        
        with torch.inference_mode():
            output=model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                use_cache=True
            )
        
        generated_ids=output[:,inputs['input_ids'].shape[1]:]
        response_tail=processor.decode(generated_ids[0],skip_special_tokens=True)
        
        full_response="This is "+response_tail.strip()
        
        diagnosis,grade=parse_grade_from_text(full_response)
        color=get_severity_color(grade)
        recommendation=get_recommendation(grade)
        
        diagnosis_html=f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h2 style="color: white; margin: 0; text-shadow: 1px 1px 2px black;">{diagnosis}</h2>
        </div>
        """
        
        recommendation_html=f"""
        <div style="background-color: #2d2d2d; padding: 15px; border-left: 5px solid {color}; border-radius: 5px; margin-top: 10px;">
            <h4 style="color: #e0e0e0; margin-top: 0;">Clinical Recommendation:</h4>
            <p style="color: #b0b0b0; margin-bottom: 0;">{recommendation}</p>
        </div>
        """
        
        response_html=f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 5px; margin-top: 10px; font-family: monospace;">
            <strong style="color: #00bcd4;">Model Output:</strong><br>
            <span style="color: #cccccc;">{full_response}</span>
        </div>
        """
        
        return diagnosis_html,recommendation_html,response_html
        
    except Exception as e:
        error_html=f'<div style="padding: 20px; background-color: #b00020; color: white; border-radius: 10px;"><strong>Error:</strong> {str(e)}</div>'
        return error_html,"",""

theme=gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#121212",
    block_background_fill="#1e1e1e",
    block_border_width="0px",
    input_background_fill="#2d2d2d",
    body_text_color="#e0e0e0",
    block_label_text_color="#e0e0e0",
    block_title_text_color="#e0e0e0",
    button_secondary_text_color="#e0e0e0"
)

with gr.Blocks(title="RetinAI",theme=theme) as demo:
    with gr.Row(variant="panel"):
        with gr.Column(scale=1,min_width=100):
            if os.path.exists(LOGO_PATH):
                gr.Image(LOGO_PATH,show_label=False,show_download_button=False,container=False,height=80)
            else:
                gr.Markdown("# 👁️ RetinAI")
        with gr.Column(scale=4):
            gr.Markdown("""
                # Diabetic Retinopathy Screening
                **AI-Powered Analysis for Early Detection**
                
                 *Model will load automatically on first analysis (2-5 minutes)*
                """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input=gr.Image(label="Fundus Image Input",type="pil",height=450,sources=["upload","clipboard"])
            with gr.Row():
                classify_btn=gr.Button("RUN ANALYSIS",variant="primary",size="lg")
                clear_btn=gr.Button("Clear",variant="secondary",size="lg")
            gr.Markdown("### Supported Grades:\n* 0: No DR\n* 1: Mild\n* 2: Moderate\n* 3: Severe\n* 4: Proliferative")

        with gr.Column(scale=1):
            diagnosis_output=gr.HTML(label="Diagnosis")
            recommendation_output=gr.HTML(label="Action Plan")
            response_output=gr.HTML(label="Raw Output")

    classify_btn.click(fn=classify_retinal_image,inputs=image_input,outputs=[diagnosis_output,recommendation_output,response_output])
    clear_btn.click(lambda:(None,"","",""),outputs=[image_input,diagnosis_output,recommendation_output,response_output])

if __name__=="__main__":
    print("\nStarting Gradio Interface...")
    print("Model will load on first inference request")
    print("Please wait for the interface URL to appear\n")
    Thread(target=load_model, daemon=True).start()
    
    demo.launch(server_name="0.0.0.0",server_port=7860,share=True)