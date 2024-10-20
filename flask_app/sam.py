import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import base64
from PIL import Image
from io import BytesIO
import warnings

def segment(base64_image):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    try:
        image_data = base64.b64decode(base64_image)
        image_pil = Image.open(BytesIO(image_data))
    except Exception as e:
        return {"error": f"Failed to decode and open image: {str(e)}"}

    try:
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        sam_checkpoint = "flask_app/SAMcheckpoints/model.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = sam_model_registry["vit_l"](checkpoint=sam_checkpoint).to(device)  # Using vit/L model

        mask_generator = SamAutomaticMaskGenerator(
            predictor,
            pred_iou_thresh=0.93,
            min_mask_region_area=1500,
        )

        sam2_result = mask_generator.generate(image_bgr)

        masks = [mask['segmentation'] for mask in sorted(sam2_result, key=lambda x: x['area'], reverse=True)]

        polygons = []
        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) >= 3:
                    polygon = contour.reshape(-1, 2).tolist()
                    polygons.append(polygon)

    except Exception as e:
        return {"error": f"Failed during segmentation process: {str(e)}"}

    return masks, polygons
