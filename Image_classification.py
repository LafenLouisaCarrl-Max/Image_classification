import numpy as np
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import gradio as gr


# ─────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────

print("Loading vision model...")
image_classifier = pipeline("image-classification",model="google/vit-base-patch16-224",device=-1)
print("Vision model ready!")


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────
def preprocess_image(image):
   """Prepare image for classification.
   - Convert to RGB (in case it's grayscale or RGBA)
   - Resize to a standard size for display"""
# Convert to RGB (removes alpha channel if PNG, handles grayscale)
   image = image.convert("RGB")
# Resize for display only (the pipeline handles model-required sizing)
   display_image = image.copy()
   display_image.thumbnail((512, 512)) # Max 512x512 while keeping aspect ratio
   return image, display_image


# ─────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────
def inspect_component(image, top_k):
  """Classifies a component image.
  Args:
    image: PIL Image from Gradio upload
    top_k: Integer — how many predictions to return
  Returns:
    label_dict: Dict for gr.Label component
    analysis_text: String summary of the analysis"""
  if image is None:
    return {}, "Please upload an image."
  # Preprocess
  image, _ = preprocess_image(image)

  # Classify
  predictions = image_classifier(image, top_k=int(top_k))

  # Convert to Gradio Label format: {label: score}
  label_dict = {pred["label"]: pred["score"] for pred in predictions}

  # Generate analysis text
  top_pred = predictions[0]
  top_label = top_pred["label"].replace("_", " ").title()
  top_score = top_pred["score"]

  if top_score > 0.9:
     confidence_text = "Very High"
  elif top_score > 0.7:
     confidence_text = "High"
  elif top_score > 0.5:
     confidence_text = "Moderate"
  else:
     confidence_text = "Low — consider re-capturing the image"
  analysis = (
    f"Top Classification: {top_label}\n"
    f"Confidence: {top_score:.1%} ({confidence_text})\n\n"
    f"Note: This model is pretrained on general images (ImageNet).\n"
    f"Fine-tune on your PCB defect images for production use."
    )
  return label_dict, analysis


# ─────────────────────────────────────────────────────────────────
# GRADIO BLOCKS INTERFACE (more flexible than gr.Interface)
# ─────────────────────────────────────────────────────────────────


with gr.Blocks(theme=gr.themes.Monochrome(), title="PCB Inspector") as app:
   gr.Markdown("""# 🔬 PCB & Component Visual Inspector### Electronics Quality Control — Upload a component photo for AI analysis""")
   with gr.Row():
      with gr.Column():
         image_input = gr.Image(
            type= "pil",
            label= "📷 Upload Component Image",
            height= 300,
            sources= ["upload", "webcam", "clipboard"]
         )
         # Slider for how many results to show
         topk_slider = gr.Slider(
            minimum = 1,
            maximum = 10,
            value = 3,
            step = 1,
            label = "Number of Predictions to Show"
         ) 
         # Submit button
         submit_btn = gr.Button("🔍 Analyze Component", variant="primary")

      with gr.Column():
         # gr.Label for classification results — renders as confidence bar chart
         label_output = gr.Label(label="🏷 Classification Results",num_top_classes=10)  

         # Text summary
         analysis_output = gr.Textbox(
            label="📊 Analysis Summary", 
            lines=6,
            interactive=False) 
         
         # Connect button
         submit_btn.click(
            fn=inspect_component,
            inputs=[image_input, topk_slider],
            outputs=[label_output, analysis_output])
         
         # Also trigger on image upload (no need to click button)
         image_input.change(
            fn=inspect_component,
            inputs=[image_input, topk_slider],
            outputs=[label_output, analysis_output])
         
         gr.Markdown("""
                     ---
                     **African Engineering Use Cases:**
                     - 🇨🇲 Cameroon — PCB quality control for electronics assembly SMEs
                     - 🇳🇬 Nigeria — Component identification for repair shops in Alaba market
                     - 🇰🇪 Kenya — Drone inspection of solar panel installations
                     - 🇬🇭 Ghana — Manufacturing defect detection for export-grade electronics
                     - 🇿🇦 South Africa — AI-powered inventory management for electronics retailers
                     """)
         
app.launch(
   share=True,
   inbrowser = True, 
   server_port=7861)         