import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Custom CSS for animated cards, colorful sectioned background, and layout
st.markdown("""
    <style>
    html, body, .stApp {
        height: 100%;
        margin: 0;
        background: linear-gradient(180deg, #f8b500 0%, #fceabb 33%, #ff9a9e 66%, #fad0c4 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .main {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    }
    .card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
        margin-top: 2rem;
    }
    .card {
        background: #ffffff10;
        border: 2px solid #ffffff33;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        width: 250px;
        height: 300px;
        perspective: 1000px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    .card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        transition: transform 0.8s;
        transform-style: preserve-3d;
    }
    .card:hover .card-inner {
        transform: rotateY(180deg);
    }
    .card-front, .card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 20px;
        padding: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        flex-direction: column;
    }
    .card-front {
        background: linear-gradient(to top left, #36D1DC, #5B86E5);
        color: white;
    }
    .card-front i {
        font-size: 40px;
        margin-bottom: 10px;
    }
    .card-back {
        background: linear-gradient(to bottom right, #ffe29f, #ffa99f);
        color: #333;
        transform: rotateY(180deg);
    }
    .apply-button {
        padding: 10px 20px;
        border: none;
        border-radius: 12px;
        background-color: #2BC0E4;
        color: white;
        font-weight: bold;
        cursor: pointer;
        margin-top: 10px;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Image processing functions
def apply_smoothing(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_contrast_stretching(image):
    in_min = np.percentile(image, 5)
    in_max = np.percentile(image, 95)
    stretched = (image - in_min) * (255 / (in_max - in_min))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.magnitude(sobelx, sobely)
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    return edge

def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1
    log_image = np.log(image) * (255 / np.log(256))
    return np.clip(log_image, 0, 255).astype(np.uint8)

# Streamlit UI
st.title("🎨 Interactive Image Filter App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_column_width=True)

    st.markdown("<h3 style='text-align: center;'>Choose a Filter</h3>", unsafe_allow_html=True)

    filters = [
        ("Smoothing", "fa-solid fa-water", apply_smoothing),
        ("Sharpening", "fa-solid fa-star", apply_sharpening),
        ("Contrast Stretching", "fa-solid fa-adjust", apply_contrast_stretching),
        ("Edge Detection (Sobel)", "fa-solid fa-vector-square", apply_edge_detection),
        ("Logarithmic Transformation", "fa-solid fa-chart-line", apply_log_transformation)
    ]

    st.markdown("<div class='card-container'>", unsafe_allow_html=True)
    for name, icon, func in filters:
        st.markdown(f"""
        <div class="card">
          <div class="card-inner">
            <div class="card-front">
              <i class="{icon}"></i>
              {name}
            </div>
            <div class="card-back">
              Click to apply filter
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Apply {name}"):
            result = func(image_np)
            display_mode = "RGB" if len(result.shape) == 3 else "GRAY"
            st.image(result, caption=f"{name} Result", use_column_width=True, channels=display_mode)
    st.markdown("</div>", unsafe_allow_html=True)

    selected_filter = st.selectbox("Or choose a filter from here:", [f[0] for f in filters])

    if st.button("Apply Selected Filter"):
        func = {f[0]: f[2] for f in filters}[selected_filter]
        result = func(image_np)
        display_mode = "RGB" if len(result.shape) == 3 else "GRAY"
        st.image(result, caption=f"{selected_filter} Result", use_column_width=True, channels=display_mode)
