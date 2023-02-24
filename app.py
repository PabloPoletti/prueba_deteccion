import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
#import cv2
import av #strealing video library
import numpy as np
import torch
from time import time
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from PIL import Image

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


image = Image.open('Yolov8_banner.jpg')
st.image(image)

st.header('Object Detection Example using YOLOv8')


st.sidebar.title("Configurations")
with st.sidebar:
    confi = st.slider('Confidence Level', 0.00, 1.00, 0.25)


########################################################################

#define gpu or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

#declaring the model (Use cache for reduce rss computational of Streamlit)
@st.cache(allow_output_mutation=True)
def load_model():
    return YOLO("yolov8n.pt") # Change to Nano Version, for reduce rss use

#model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
model = load_model()
model.fuse()

#import the class names    
CLASS_NAMES_DICT = model.model.names

#define the box_annotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)

#define the rtc config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

dict_esp = {0: 'persona',1: 'bicicleta',2: 'coche',3: 'motocicleta',4: 'avión',5: 'autobús',6: 'tren',7: 'camión',8: 'barco',9: 'semáforo',
10: 'boca de incendios',11: 'señal de alto',12: 'parquímetro',13: 'banco',14: 'pájaro',15: 'gato',16: 'perro',17: 'caballo',18: 'oveja',
19: 'vaca',20: 'elefante',21: 'oso',22: 'cebra',23: 'jirafa',24: 'mochila',25: 'paraguas',26: 'bolso',27: 'corbata',28: 'maleta',
29: 'frisbee',30: 'esquís',31: 'tabla de snowboard',32: 'pelota deportiva',33: 'cometa',34: 'bate de béisbol',35: 'guante de béisbol',
36: 'monopatín',37: 'tabla de surf',38: 'raqueta de tenis',39: 'botella',40: 'copa de vino',41: 'taza',42: 'tenedor',43: 'cuchillo', 
44: 'cuchara',45: 'cuenco',46: 'plátano',47: 'manzana',48: 'sándwich',49: 'naranja',50: 'brócoli',51: 'zanahoria',52: 'perro caliente',
53: 'pizza',54: 'rosquilla',55: 'pastel',56: 'silla',57: 'sofá', 58: 'planta en maceta',59: 'cama',60: 'mesa de comedor',61: 'baño',
62: 'televisión',63: 'portátil',64: 'ratón',65: 'remoto',66: 'teclado',67: 'celular',68: 'microondas',69: 'horno',70: 'tostadora',
71: 'lavabo',72: 'refrigerador',73: 'libro',74: 'reloj',75: 'jarrón',76: 'tijeras',77: 'oso de peluche',78: 'secador de pelo',
79: 'cepillo de dientes'}

with st.sidebar:
    lang_label = st.selectbox(
        'Select Label Language',
        ('English', 'Spanish'))

if lang_label == 'English':
    lenguaje = CLASS_NAMES_DICT
else:
    lenguaje = dict_esp


def predict(frame):
    results = model(frame, conf=confi)
    return results


def plot_bboxes(results, frame):
    xyxys = []
    confidences = []
    class_ids = []
    # Extract detections for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        if class_id == 0:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
    # Setup detections for visualization
    detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                )
    # Format custom labels
    labels = [f"{lenguaje[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections]
    # Annotate and display frame
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    frame = np.fliplr(frame)
    return frame


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = predict(img)
        img = plot_bboxes(results,img)
        # fps = 1/np.round(end_time - start_time, 2)
        # cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        img = np.fliplr(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)





st.subheader("80 different classes that this pre-trained model detects")

image = Image.open('Coco_Classes.png')

st.image(image, caption='COCO Classes')

st.write("The full github code is [here](https://github.com/PabloPoletti/prueba_deteccion)")
