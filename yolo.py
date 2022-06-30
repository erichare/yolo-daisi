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

    try:
        image = Image.open(image)
    except Exception as _:
        pass

    # Now save the bytes of the image to a BytesIO
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize the image to the max dimensions
    new_width  = 768
    new_height = int(new_width * image.size[1] / image.size[0])
    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Write the image
    source = os.path.join(tmpdir, source_name)
    image.save(source)
        
    # Streamlit
    inferer = Inferer(source, weights, device="", yaml="coco.yaml", 
                      img_size=new_width, half=False)
    inferer.infer(conf_thres=.25, iou_thres=.45, classes=None, 
                  agnostic_nms=False, max_det=1000, save_dir=tmpdir, 
                  save_txt=True, save_img=True, hide_labels=False, hide_conf=False)
    
    result_img = Image.open(os.path.join(tmpdir, os.path.basename(source)))
    result_img.load()

    return image, result_img

if __name__ == "__main__":
    st.title("Yolo V6 Model")

    st.write("This Daisi allows you to provide an image, and one of the most advanced Object Detection algorithms available will try to classify it for you. Upload your data to get started!")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an Image", type=["png","jpg","jpeg"])

    if not uploaded_file:
        file_name = "busystreet.png"
    else:
        file_name = uploaded_file.name

    with st.expander("Show PyDaisi Code"):
        st.markdown('## Calling with PyDaisi')
        st.markdown(f"""
        ```python
        import pydaisi as pyd
        from PIL import Image

        yolo_object_detection = pyd.Daisi("erichare/YOLO Object Detection")

        img = Image.open("{file_name}")
        img.load()

        original_image, result_image = yolo_object_detection.yolo(img).value

        original_image.show()
        result_image.show()
        ```
        """)

    original_image, result_image = yolo(uploaded_file)

    st.image(original_image, caption='Original Image')
    st.image(result_image, caption='Objects Detected')
