import os
import streamlit as st
import tempfile
import uuid
import base64

from PIL import Image

from yolov6.core.inferer import Inferer

weights = "yolov6s.pt"
my_uuid = str(uuid.uuid4())
tmpdir = os.path.join(tempfile.gettempdir(), my_uuid)
tmp_path = os.path.join(tmpdir, "labels")
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

def yolo(source="cat.jpeg"):
    if not source:
        source = "cat.jpeg"

    print(my_uuid)
    print(source)
    inferer = Inferer(source, weights, device="", yaml="coco.yaml", 
                      img_size=640, half=False)
    inferer.infer(conf_thres=.25, iou_thres=.45, classes=None, 
                  agnostic_nms=False, max_det=1000, save_dir=tmpdir, 
                  save_txt=True, save_img=True, hide_labels=False, hide_conf=False)
    
    with open(os.path.join(tmpdir, os.path.basename(source)), 'rb') as f:
        im = base64.b64encode(f.read()).decode()

    return [{"type": "image", "label": "yolov6", "data":  {"alt": "yolov6 result", "src": "data:image/png;base64, " + im}}]

if __name__ == "__main__":
    st.title("Test")