import pytest
import base64
from PIL import Image
import io
import sd
import clip_basic
import clip_advanced
import sam

@pytest.fixture
def sample_image():
    prompt = "A beautiful landscape"
    num_inference_steps = 50
    guidance_scale = 7.5
    height = 512
    width = 512

    # Create a sample image and encode it to base64
    image = Image.new('RGB', (width, height), color='blue')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "img_b64": img_b64
    }

def test_stable_diffusion(sample_image):
    result = sd.stable_diff(
        sample_image["prompt"],
        sample_image["num_inference_steps"],
        sample_image["guidance_scale"],
        sample_image["height"],
        sample_image["width"]
    )
    assert "error" not in result

def test_segmentation(sample_image):
    result = sam.segment(sample_image["img_b64"])
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)

def test_basic_clip_analysis(sample_image):
    text_descriptions = ["landscape", "portrait", "abstract"]
    result = clip_basic.clip_img(sample_image["img_b64"], text_descriptions)
    assert isinstance(result, dict)
    assert len(result) == len(text_descriptions)

def test_extended_clip_analysis(sample_image, tmpdir):
    mock_file = tmpdir.join("temp_objects.txt")
    mock_file.write("\n".join(["object1", "object2", "object3", "object4", "object5"]))
    result = clip_advanced.clip_img(sample_image["img_b64"])
    assert isinstance(result, dict)
    assert len(result) <= 10