import unittest
import sd, sam, clip_basic, clip_advanced

class TestFunctions(unittest.TestCase):

    def test_stable_diff_errors(self):
        #empty prompt
        result = sd.stable_diff("", 50, 7.5, 512, 512)
        self.assertIn("error", result)
        self.assertIn("Invalid prompt. The prompt cannot be empty.", result["error"])
        
        #witspace prompt
        result = sd.stable_diff("   ", 50, 7.5, 512, 512)
        self.assertIn("error", result)
        self.assertIn("Invalid prompt. The prompt cannot be empty.", result["error"])
        
        #invalid height
        result = sd.stable_diff("A beautiful sunset", 50, 7.5, 0, 512)
        self.assertIn("error", result)
        self.assertIn("Invalid dimensions. Height and width must be greater than zero.", result["error"])
        
        #invalid width
        result = sd.stable_diff("A beautiful sunset", 50, 7.5, 512, 0)
        self.assertIn("error", result)
        self.assertIn("Invalid dimensions. Height and width must be greater than zero.", result["error"])


    def test_segment_errors(self):
        #invalid base64 img
        result = sam.segment("invalid_base64_string")
        self.assertIn("error", result)
        self.assertIn("Failed to decode and open image", result["error"])

        #incompatible datatype
        result = sam.segment(None)
        self.assertIn("error", result)
        self.assertIn("Failed to decode and open image", result["error"])

    def test_clip_img_errors_with_text(self):
        #empty image
        result = clip_basic.clip_img("", ["A cat", "A dog"])
        self.assertIn("error", result)
        self.assertIn("Failed to decode image", result["error"])

        #invalid image
        result = clip_basic.clip_img("invalid_base64_string", ["A cat", "A dog"])
        self.assertIn("error", result)
        self.assertIn("Failed to decode image", result["error"])

    def test_clip_img_errors_without_text(self):
        #empty image
        result = clip_advanced.clip_img("")
        self.assertIn("error", result)
        self.assertIn("Failed to decode image", result["error"])

        #util function not present
        try:
            result = clip_advanced.clip_img("valid_base64_string")
        except Exception as e:
            self.assertIn("error", str(e))

        #model loading failure
        try:
            result = clip_advanced.clip_img("valid_base64_string")
        except Exception as e:
            self.assertIn("error", str(e))

if __name__ == '__main__':
    unittest.main()
