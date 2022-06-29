import scipy
import os
import streamlit as st
import numpy as np
import tempfile
import uuid
import base64

from scipy import misc
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
    source_name = str(uuid.uuid4()) + ".jpg"
    if hasattr(image, "name"):
        source_name = image.name

    if image is None:
        image = scipy.misc.face(gray=True).astype(np.float32)

    if type(image) == np.ndarray:
        img_crop_pil = Image.fromarray(image)
        if img_crop_pil.mode != 'RGB':
            img_crop_pil = img_crop_pil.convert('RGB')

        image = BytesIO()
        img_crop_pil.save(image, format="jpeg")        
        
    # Streamlit
    if type(image) == BytesIO:
        source_path = os.path.join(tmpdir, source_name)
        with open(source_path, mode='wb') as f:
            f.write(image.getbuffer())
        source = source_path
    elif type(image) == str:
        source = image

    print(my_uuid)
    print(source)
    inferer = Inferer(source, weights, device="", yaml="coco.yaml", 
                      img_size=1280, half=False)
    inferer.infer(conf_thres=.25, iou_thres=.45, classes=None, 
                  agnostic_nms=False, max_det=1000, save_dir=tmpdir, 
                  save_txt=True, save_img=True, hide_labels=False, hide_conf=False)
    
    with open(os.path.join(tmpdir, os.path.basename(source)), 'rb') as f:
        im = base64.b64encode(f.read()).decode()

    return [{"type": "image", "label": "yolov6", "data":  {"alt": "yolov6 result", "src": "data:image/png;base64, " + im}}]

if __name__ == "__main__":
    st.title("Yolo V6 Model")

    st.write("This Daisi allows you to provide an image, and one of the most advanced Object Detection algorithms available will try to classify it for you. Upload your data to get started!")
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file")

    print(uploaded_file)
    res = yolo(uploaded_file)

    st.markdown('<img src="' + res[0]["data"]["src"] + '"/>', unsafe_allow_html=True)
