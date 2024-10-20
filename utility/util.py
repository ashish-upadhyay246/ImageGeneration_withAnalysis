import supervision as sv
import cv2
import matplotlib.pyplot as plt
import numpy as np

#pop up to show the annotated image along with masks
def annot_img(sam_result, image_bgr):
    #annotating the original image (creating an overlay over the original image) with the generated masks
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['Source Image', 'Segmented Image']
    )

#pop up to show all the polygons identified from the masks
def vis_polygons(masks,polygons):
    num_masks = len(masks)
    grid_size = (min(num_masks, 4), (num_masks + 3) // 4)
    plt.figure(figsize=(8, 8))
    for i, mask in enumerate(masks):
        mask_color = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for polygon in polygons:
            if len(polygon) >= 3:
                cv2.polylines(mask_color, [np.array(polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.imshow(mask_color)
        plt.title(f'Mask {i + 1}', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()