import base64
import requests
import streamlit as st
import modules
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")

if 'generated' not in st.session_state:
    st.session_state.generated = False
    st.session_state.generated_image_b64 = None
    st.session_state.generated_image = None

st.title("Image Generation, Analysis, and Segmentation Pipeline")
col1, col2, col3 = st.columns([1, 2, 2])
col1.subheader("Input")

with col1:
    prompt = st.text_input("Enter a prompt for image generation:")
    labels_to_test = st.text_input("Enter the objects to be analyzed (e.g., a horse, a chair, etc.):")
    num_inference_steps = st.slider("Number of Inference Steps:", min_value=30, max_value=75, step=2)
    guidance_scale = st.slider("Guidance Scale (CFG):", min_value=1.0, value=7.5, max_value=10.0, step=0.5)
    height = st.slider("Height:", min_value=128, value=512, max_value=768, step=128)
    width = st.slider("Width:", min_value=128, value=512, max_value=768, step=128)

if (st.button("Generate Image") and prompt):
    url = "http://localhost:5000/generate"
    payload = {
        "prompt": prompt,
        "labels": labels_to_test,
        "samples": num_inference_steps,
        "cfg": guidance_scale,
        "height": height,
        "width": width
    }
    with col2:
        with st.spinner('Generating image. Please wait...'):
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

                st.session_state.generated = True
                st.session_state.generated_image_b64 = result['generated_image']
                st.session_state.generated_image = Image.open(BytesIO(base64.b64decode(st.session_state.generated_image_b64)))
                
                if modules.is_black_image(st.session_state.generated_image):
                    st.warning("The generated image is a black image (possibly NSFW). Please provide different input.")
                else:
                    with col2:
                        st.image(st.session_state.generated_image, use_column_width=False, width=512)

                    with col3:
                        st.subheader("CLIP analysis on the provided keywords")
                        modules.print_clip(result)

            except requests.exceptions.RequestException as req_e:
                st.error(f"Request error: {req_e}")
            except ValueError as json_e:
                st.error(f"Failed to decode JSON response: {json_e}")
            except Exception as e:
                st.error(f"Error: {e}")


if st.session_state.generated_image_b64:
    if st.button("Analyze and Segment"):
        url = "http://localhost:5000/analyze"
        with col2:
            with st.spinner('Analyzing image and generating segments. Please wait...'):
                try:
                    payload = {
                        "image": st.session_state.generated_image_b64,
                    }

                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()

                    generated_image_b64 = result['generated_image']
                    generated_image = Image.open(BytesIO(base64.b64decode(generated_image_b64)))
                    
                    with col2:
                        st.image(generated_image, caption="Analyzed Image", use_column_width=False, width=512)
                    
                    with col3:
                        st.subheader("CLIP Analysis of the image on extended keywords")
                        modules.print_clip(result)
                    
                    masks = result['basic_segmentation']['masks']
                    polygons = result['basic_segmentation']['polygons']
                    
                    with col3:
                        modules.plot_masks(masks, generated_image)
                        modules.plot_polygons(polygons, generated_image)

                except requests.exceptions.RequestException as req_e:
                    st.error(f"Request error: {req_e}")
                except ValueError as json_e:
                    st.error(f"Failed to decode JSON response: {json_e}. Response: {response.text}")
                except KeyError as key_e:
                    st.error(f"Missing key in response: {key_e}")
                except Exception as e:
                    st.error(f"Error: {e}")