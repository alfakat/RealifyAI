import torch
from PIL import Image
import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor, CLIPProcessor, CLIPModel

"""Load models"""
# Path to model can be replaced by local one
deepfake_model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
deepfake_processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
clip_model_path = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(clip_model_path)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
clip_model.eval()
deepfake_model.eval()

clip_labels = ["photo of a real person or scene", "synthetic or AI generated image"]


def hybrid_classifier(img: Image.Image):
    df_inputs = deepfake_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        df_outputs = deepfake_model(**df_inputs)
    df_probs = torch.nn.functional.softmax(df_outputs.logits, dim=1)
    df_real_score = df_probs[0][deepfake_model.config.label2id["Realism"]].item()
    df_fake_score = df_probs[0][deepfake_model.config.label2id["Deepfake"]].item()

    clip_inputs = clip_processor(text=clip_labels, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        clip_probs = torch.softmax(clip_outputs.logits_per_image, dim=1).squeeze()
    clip_real_score = clip_probs[0].item()
    clip_fake_score = clip_probs[1].item()

    final_real_score = (0.3 * df_real_score + 0.7 * clip_real_score)
    final_fake_score = (0.3 * df_fake_score + 0.7 * clip_fake_score)

    decision = "Generated"
    if abs(final_real_score - final_fake_score) > 0.3:
        decision = "Real" if final_real_score > final_fake_score else decision

    return decision


"""Preset and launch Gradio """
iface = gr.Interface(
    fn=hybrid_classifier,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Final Decision"),
    title="RealifyAI",
    description="Figure out if image real or generated",
    examples=[["examples/image00.jpg"],
        ["examples/image01.jpg"],
        ["examples/image02.jpg"]])

if __name__ == "__main__":
    iface.launch()
