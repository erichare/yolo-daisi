import os
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import uuid

from PIL import Image

from yolov6.core.inferer import Inferer

weights = "yolov6s.pt"
my_uuid = str(uuid.uuid4())

tmpdir = os.path.join(tempfile.gettempdir(), my_uuid)
tmp_path = os.path.join(tmpdir, "labels")

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

def yolo(image: np.ndarray=None, return_type: list=["Image", "Labels"]):
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

    # Get the boolean flags
    save_img = "Image" in return_type
    save_txt = "Labels" in return_type

    # Write the image
    source = os.path.join(tmpdir, source_name)
    image.save(source)
        
    # Streamlit
    inferer = Inferer(source, weights, device="", yaml="coco.yaml", 
                      img_size=new_width, half=False)
    inferer.infer(conf_thres=.25, iou_thres=.45, classes=None, 
                  agnostic_nms=False, max_det=1000, save_dir=tmpdir, 
                  save_txt=save_txt, save_img=save_img, hide_labels=False, hide_conf=False)
    
    if save_img:
        yolo_result = Image.open(os.path.join(tmpdir, os.path.basename(source)))
        yolo_result.load()

    if save_txt:
        source_base = os.path.basename(source)
        source_root, file_extension = os.path.splitext(source_base)
        label_result = pd.read_csv(os.path.join(tmpdir, "labels", source_root + ".txt"), sep=" ", header=None)

    if len(return_type) == 2:
        return yolo_result, label_result
    elif save_img:
        return yolo_result
    elif save_txt:
        return label_result
    else:
        return None

if __name__ == "__main__":
    st.title("Yolo V6 Model")
    st.write("This Daisi allows you to provide an image, and one of the most advanced Object Detection algorithms available will try to classify it for you. Upload your data to get started!")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an Image", type=["png","jpg","jpeg"])
        return_types = st.multiselect("Select Return Type", ["Image", "Labels"], ["Image", "Labels"])

    if not uploaded_file:
        file_name = "busystreet.png"
    else:
        file_name = uploaded_file.name

    with st.expander("Inference with PyDaisi"):
        st.markdown(f"""
        ```python
        import pydaisi as pyd
        from PIL import Image

        yolo_object_detection = pyd.Daisi("erichare/YOLO Object Detection")

        img = Image.open("{file_name}")
        img.load()

        yolo_result = yolo_object_detection.yolo(img).value
        yolo_result.show()
        ```
        """)

    final_result = yolo(uploaded_file, return_type=return_types)
    if type(final_result) == tuple:
        yolo_result, label_result = final_result
    elif "Image" in return_types:
        yolo_result = final_result
    elif "Labels" in return_types:
        label_result = final_result
    
    if "Image" in return_types:
        st.image(yolo_result, caption='Objects Detected')
    if "Labels" in return_types:
        st.table(label_result)
