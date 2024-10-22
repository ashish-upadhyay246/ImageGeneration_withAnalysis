import uuid
import logging
from flask import Flask, request, jsonify
import sd, clip_basic, clip_advanced, sam

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#endpoint for generating the image and providing basic CLIP analysis
@app.route('/generate', methods=['POST'])
def generate_img():
    logger.debug("Entered generate_img() method")
    try:
        data = request.json

        #validating endpoint
        if not data or 'prompt' not in data or not isinstance(data['prompt'], str):
            logger.warning("Validation error: 'prompt' is required and must be a string.")
            return jsonify({"error": "'prompt' is required and must be a string."}), 400

        labels = data.get('labels', '')
        labels_list = [label.strip() for label in labels.split(',')] if labels else []
        num_inference_steps = data.get('samples', 30)
        guidance_scale = data.get('cfg', 7.5)
        h = data.get('height', 512)
        w = data.get('width', 768)

        #validating endpoint
        for field, value in [('samples', num_inference_steps), ('cfg', guidance_scale), ('height', h), ('width', w)]:
            if not isinstance(value, (int, float)) or value <= 0:
                logger.warning(f"Validation error: '{field}' must be a positive number.")
                return jsonify({"error": f"'{field}' must be a positive number."}), 400

        logger.info(f"Received request for image generation with prompt: {data['prompt']}, labels: {labels_list}")

        image_b64 = sd.stable_diff(data['prompt'], num_inference_steps, guidance_scale, h, w)
        if isinstance(image_b64,dict):
            logger.debug("Could not generate image. Problem occured while either loading the pipeline or saving the image.")
            print(image_b64)
        else:
            logger.debug("Stable diffusion image generation completed")

        clip_scores = clip_basic.clip_img(image_b64, labels_list)
        logger.debug("Basic CLIP analysis completed")

        response = {
            "request_id": str(uuid.uuid4()),
            "generated_image": image_b64,
            "clip_analysis": {
                "concepts": list(clip_scores.keys()),
                "confidence_scores": clip_scores
            }
        }
        logger.info("Image generation and analysis successful. Displaying the image, please wait...")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error occured in generate_img() function: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


#endpoint for providing extended CLIP analysis and image segmentation
@app.route('/analyze', methods=['POST'])
def analyze_image():
    logger.debug("Entered analyze_image() method")
    try:
        data = request.json

        #validating endpoint
        if not data or 'image' not in data or not isinstance(data['image'], str):
            logger.warning("Validation error: 'image' is required and must be a string.")
            return jsonify({"error": "'image' is required and must be a string."}), 400

        img_b64 = data["image"]
        logger.info("Received request for image analysis")

        clip_scores = clip_advanced.clip_img(img_b64)
        logger.debug("Extended CLIP analysis completed")

        masks, polygons = sam.segment(img_b64)
        logger.debug("Image segmentation completed")

        masks_list = [mask.tolist() for mask in masks]

        response = {
            "request_id": str(uuid.uuid4()),
            "generated_image": img_b64,
            "clip_analysis": {
                "concepts": list(clip_scores.keys()),
                "confidence_scores": clip_scores
            },
            "basic_segmentation": {
                "masks": masks_list,
                "polygons": polygons
            }
        }
        logger.info("Image analysis and segmentation successful. Displaying the masks and ROI, please wait...")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error occured in analyze_image() method: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
