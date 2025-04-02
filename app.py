import cv2
import numpy as np
import torch
import torchvision.transforms as T
import gradio as gr
from torchvision.models.segmentation import deeplabv3_resnet101

# Load the DeepLabV3 model for segmentation
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Define preprocessing transformations
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def segment_person(image):
    """Segments a person from an image using DeepLabV3."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)["out"][0]
    mask = output.argmax(0).byte().numpy()

    # Extract only the person (Class 15 in DeepLabV3)
    person_mask = (mask == 15).astype(np.uint8) * 255
    segmented = cv2.bitwise_and(img, img, mask=person_mask)

    return segmented, person_mask

def insert_into_stereo(segmented, mask, left_img, right_img, depth="medium"):
    """Inserts segmented person into stereoscopic images at a given depth."""
    disparities = {"close": 30, "medium": 15, "far": 5}
    shift = disparities.get(depth, 15)

    # Find bounding box of person
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return left_img, right_img

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    person_crop = segmented[y_min:y_max, x_min:x_max]
    mask_crop = mask[y_min:y_max, x_min:x_max]

    x_pos_left, x_pos_right = 100, 100 + shift  # Adjust positions

    # Insert into left and right images
    left_img[y_min:y_max, x_pos_left:x_pos_left + (x_max - x_min)] = cv2.bitwise_and(
        left_img[y_min:y_max, x_pos_left:x_pos_left + (x_max - x_min)], 
        left_img[y_min:y_max, x_pos_left:x_pos_left + (x_max - x_min)], 
        mask=cv2.bitwise_not(mask_crop)
    ) + person_crop

    right_img[y_min:y_max, x_pos_right:x_pos_right + (x_max - x_min)] = cv2.bitwise_and(
        right_img[y_min:y_max, x_pos_right:x_pos_right + (x_max - x_min)], 
        right_img[y_min:y_max, x_pos_right:x_pos_right + (x_max - x_min)], 
        mask=cv2.bitwise_not(mask_crop)
    ) + person_crop

    return left_img, right_img

def create_anaglyph(left_img, right_img):
    """Creates an anaglyph 3D image from left and right stereo images."""
    left_red = left_img.copy()
    left_red[:, :, 1:] = 0  # Keep only red channel

    right_cyan = right_img.copy()
    right_cyan[:, :, 0] = 0  # Remove red channel

    anaglyph = cv2.addWeighted(left_red, 0.5, right_cyan, 0.5, 0)
    return anaglyph

def process_image(person_image, left_image, right_image, depth):
    """Processes the input images and returns segmented, stereo, and anaglyph images."""
    segmented, mask = segment_person(person_image)
    left, right = insert_into_stereo(segmented, mask, left_image, right_image, depth)
    anaglyph = create_anaglyph(left, right)
    return segmented, left, right, anaglyph

# Gradio UI
app = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Person Image"),
        gr.Image(type="numpy", label="Left Stereo Image"),
        gr.Image(type="numpy", label="Right Stereo Image"),
        gr.Dropdown(["close", "medium", "far"], label="Depth Level")
    ],
    outputs=[
        gr.Image(type="numpy", label="Segmented Person"),
        gr.Image(type="numpy", label="Left Image with Person"),
        gr.Image(type="numpy", label="Right Image with Person"),
        gr.Image(type="numpy", label="Anaglyph Image")
    ],
    title="3D Image Composer",
    description="Upload an image of a person and place them into a 3D scene with depth perception. View with red-cyan glasses!"
)

if __name__ == "__main__":
    app.launch(share=True)
