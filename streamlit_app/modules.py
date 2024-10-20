import cv2
import streamlit as st
import numpy as np

def plot_polygons(polygons, generated_image):
    st.subheader("Visualization of polygons")
    original_image = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
    for i, polygon in enumerate(polygons):
        polygon_image = original_image.copy()
        
        #drawing polygon on the background
        cv2.fillPoly(polygon_image, [np.array(polygon, np.int32)], color=(0, 255, 0, 128))
        cv2.polylines(polygon_image, [np.array(polygon, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
        
        #converting the image to rgb to display on streamlit
        polygon_image_rgb = cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB)
        st.image(polygon_image_rgb, caption=f"Polygon {i + 1} (Coordinates: {polygon})", use_column_width=False, width=512)

def plot_masks(masks, generated_image):
    st.subheader("Masks created on the original image")
    #converting original image to format compatible with opencv
    original_image = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)

    overlay_image = original_image.copy()

    for i, mask in enumerate(masks):
        mask_uint8 = np.array(mask, dtype=np.uint8) * 255
        #generate random colors for the masks
        random_color = np.random.randint(0, 256, size=3).tolist()
        #overlay the colored masks on the original image
        overlay_image[mask_uint8 > 0] = [random_color[0], random_color[1], random_color[2]] #setting pixels of the mask to random colors

    #display original image with all masks overlayed
    annotated_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, use_column_width=False, width=512)


def print_clip(result):
    concepts = result['clip_analysis']['concepts']
    confidence_scores = result['clip_analysis']['confidence_scores']
    if(len(concepts)==0):
        st.write("No keywords were provided.")
    for concept in concepts:
        confidence = confidence_scores[concept]
        st.write(f"**{concept}**: {confidence:.2f}%")

def is_black_image(image):
    image_array = np.array(image)
    return np.all(image_array ==0)