import pytest
import requests
import json
import io
import base64
from PIL import Image

BASE_URL = "http://127.0.0.1:5000"

width=768
height=512

test_generate_payload = {
    "prompt": "A sunset over a mountain range",
    "labels": "sunset, mountain, nature",
    "samples": 25,
    "cfg": 7.5,
    "height": height,
    "width": width
}

# Create a sample image and encode it to base64
image = Image.new('RGB', (width, height), color='blue')
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


test_analyze_payload = {
    "image": img_b64
}

def test_generation():
    response = requests.post(f"{BASE_URL}/generate", json=test_generate_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "generated_image" in data
    assert "clip_analysis" in data
    assert "concepts" in data["clip_analysis"]
    assert "confidence_scores" in data["clip_analysis"]

def test_generation_with_missing_prompt():
    invalid_payload = {
        "labels": "sunset, mountain, nature"
    }
    response = requests.post(f"{BASE_URL}/generate", json=invalid_payload)
    assert response.status_code == 400

    data = response.json()
    assert "error" in data
    assert "'prompt' is required" in data["error"]

def test_analysis():
    response = requests.post(f"{BASE_URL}/analyze", json=test_analyze_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "generated_image" in data
    assert "clip_analysis" in data
    assert "concepts" in data["clip_analysis"]
    assert "confidence_scores" in data["clip_analysis"]
    assert "basic_segmentation" in data
    assert "masks" in data["basic_segmentation"]
    assert "polygons" in data["basic_segmentation"]

def test_analysis_with_missing_image():
    invalid_payload = {}
    response = requests.post(f"{BASE_URL}/analyze", json=invalid_payload)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "'image' is required" in data["error"]