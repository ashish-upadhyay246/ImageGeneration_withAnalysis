import pytest
import base64
from PIL import Image
import io
from unittest.mock import patch
import sd, sam, clip_basic, clip_advanced

image = Image.new('RGB', (128, 128), color='blue')
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
valid_base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_stable_diff_errors():
    #empty prompt
    result = sd.stable_diff("", 50, 7.5, 512, 512)
    assert "error" in result
    assert "Invalid prompt. The prompt cannot be empty." in result["error"]
    
    #whitespace prompt
    result = sd.stable_diff("   ", 50, 7.5, 512, 512)
    assert "error" in result
    assert "Invalid prompt. The prompt cannot be empty." in result["error"]
    
    #invalid height
    result = sd.stable_diff("A beautiful sunset", 50, 7.5, 0, 512)
    assert "error" in result
    assert "Invalid dimensions. Height and width must be greater than zero." in result["error"]
    
    #invalid width
    result = sd.stable_diff("A beautiful sunset", 50, 7.5, 512, 0)
    assert "error" in result
    assert "Invalid dimensions. Height and width must be greater than zero." in result["error"]

def test_segment_errors():
    #invalid base64 image
    result = sam.segment("invalid_base64_string")
    assert "error" in result
    assert "Failed to decode and open image" in result["error"]

    #incompatible datatype
    result = sam.segment(None)
    assert "error" in result
    assert "Failed to decode and open image" in result["error"]

def test_clip_img_errors_with_text():
    #testing clip_basic
    #empty image
    result = clip_basic.clip_img("", ["A cat", "A dog"])
    assert "error" in result
    assert "Failed to decode image" in result["error"]

    #invalid image
    result = clip_basic.clip_img("invalid_base64_string", ["A cat", "A dog"])
    assert "error" in result
    assert "Failed to decode image" in result["error"]

def test_clip_img_errors_without_text():
    #testing clip_advanced
    #empty image string
    result_advanced = clip_advanced.clip_img("")
    assert "error" in result_advanced
    assert "Failed to decode image" in result_advanced["error"]

    #invalid base64 image string
    result_advanced = clip_advanced.clip_img("invalid_base64_string")
    assert "error" in result_advanced
    assert "Failed to decode image" in result_advanced["error"]

    #simulating model loading failure
    with patch("clip.load", side_effect=Exception("Failed to load CLIP model")):
        result_advanced = clip_advanced.clip_img(valid_base64_string)
        assert "error" in result_advanced
        assert "Failed to load CLIP model" in result_advanced["error"]

    #inference failure test
    with patch("clip_advanced.clip.tokenize", side_effect=Exception("Failed during preprocessing or inference")):
        result_advanced = clip_advanced.clip_img(valid_base64_string)
        assert "error" in result_advanced
        assert "Failed during preprocessing or inference" in result_advanced["error"]
