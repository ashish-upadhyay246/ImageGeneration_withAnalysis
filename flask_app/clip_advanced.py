import clip
import torch
import base64
from io import BytesIO
from PIL import Image

def clip_img(img_b64):
    try:
        img_data = base64.b64decode(img_b64)
        image = Image.open(BytesIO(img_data))
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model, preprocess = clip.load("ViT-L/14", device=device)
    except Exception as e:
        return {"error": f"Failed to load CLIP model: {str(e)}"}

    try:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with open("utility/common_objects.txt", "r") as f:
            text_descriptions = [line.strip() for line in f.readlines()]

        text = clip.tokenize(text_descriptions).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(image_tensor, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    except Exception as e:
        return {"error": f"Failed during preprocessing or inference: {str(e)}"}

    confidence_scores = probs[0] * 100
    ans = {description: float(confidence) for description, confidence in zip(text_descriptions, confidence_scores)}
    sorted_ans = sorted(ans.items(), key=lambda item: item[1], reverse=True)[:5]

    return dict(sorted_ans)