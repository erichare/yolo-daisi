import scipy
import os
import streamlit as st
import numpy as np
import tempfile
import uuid
import base64

from PIL import Image
from io import BytesIO

from yolov6.core.inferer import Inferer

weights = "yolov6s.pt"
my_uuid = str(uuid.uuid4())

tmpdir = os.path.join(tempfile.gettempdir(), my_uuid)
tmp_path = os.path.join(tmpdir, "labels")

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

def yolo(image: np.ndarray=None):
    # Get a name for the image
    source_name = str(uuid.uuid4()) + ".jpg"
    if hasattr(image, "name"):
        source_name = image.name

    # If none is provided, we read the busy street image
    if image is None:
        source_name = "busystreet.png"
        image = Image.open(source_name)
        image.load()

    # If we passed in a Numpy Array, convert to image
    if type(image) == np.ndarray:
        image = Image.fromarray(image)

    # Now save the bytes of the image to a BytesIO
    if not hasattr(image, "getbuffer"):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        bytes_image = BytesIO()
        image.save(bytes_image, format="jpeg")  
        image = bytes_image      
        
    # Streamlit
    source = os.path.join(tmpdir, source_name)
    if hasattr(image, "getbuffer"):
        with open(source, mode='wb') as f:
            f.write(image.getbuffer())
    else:
        source = image

    inferer = Inferer(source, weights, device="", yaml="coco.yaml", 
                      img_size=1280, half=False)
    inferer.infer(conf_thres=.25, iou_thres=.45, classes=None, 
                  agnostic_nms=False, max_det=1000, save_dir=tmpdir, 
                  save_txt=True, save_img=True, hide_labels=False, hide_conf=False)
    
    result_img = Image.open(os.path.join(tmpdir, os.path.basename(source)))
    result_img.load()

    return result_img

if __name__ == "__main__":
    st.title("Yolo V6 Model")

    st.write("This Daisi allows you to provide an image, and one of the most advanced Object Detection algorithms available will try to classify it for you. Upload your data to get started!")
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file")

    result_image = yolo(uploaded_file)

    st.image(result_image, caption='Original Image')
    st.image(result_image, caption='Objects Detected')
