"""Identity confirmation by photo demo with Dlib models.
These models are taken from a repository on GitHub:
https://github.com/davisking/dlib-models?ysclid=lpbl2ajfs8881987614
"""
import av
import numpy as np
import cv2
import dlib
import re
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

PADDING = 50

# Models loading
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat')
face_recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

streaming_placeholder = st.empty()


class CamImageLoad:
    def __init__(self):
        self.check_button_pressed = False
        self.image_cam = None


cam_image_load = CamImageLoad()


def shape_to_np(shape, dtype="int"):
    """

    Convert shape to numpy array.

    """

    cord = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        cord[i] = (shape.part(i).x, shape.part(i).y)
    return cord


def get_descriptor(img):
    """

    Desiccator calculation.

    """

    bbox, shape = get_shape(img)
    return bbox, face_recognition.compute_face_descriptor(img, shape)


def get_shape(img):
    """

    Obtaining key points and bounding boxes.

    """

    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = shape_predictor(img, d)

        numbers = re.findall(r'\d+', str(d))
        temp_str = ', '.join(numbers)
        bbox = temp_str.split(',')

    return bbox, shape


def draw_facemask(face, bbox, shape, bbox_color):
    """

    Drawing key points and limiting frames.

    """

    points = shape_to_np(shape)

    cv2.rectangle(face, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), bbox_color, 2)

    # Lower border of the face
    for i in range(16):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)

    # Lips
    for i in range(48, 60):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)

    # Nose
    for i in range(27, 35):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)
    cv2.line(face, points[35], points[30], (0, 0, 255), 1)

    # Right eye
    for i in range(36, 41):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)
    cv2.line(face, points[36], points[41], (0, 0, 255), 1)

    # Left eye
    for i in range(42, 47):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)
    cv2.line(face, points[42], points[47], (0, 0, 255), 1)

    # Right brow
    for i in range(17, 21):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)

    # Left brow
    for i in range(22, 26):
        cv2.line(face, points[i], points[i + 1], (0, 0, 255), 1)

    x1 = max(0, int(bbox[0]) - PADDING)
    y1 = max(0, int(bbox[1]) - PADDING)
    x2 = min(face.shape[1], int(bbox[2]) + PADDING)
    y2 = min(face.shape[0], int(bbox[3]) + PADDING)

    return face[y1:y2, x1:x2]


def callback(frame: av.VideoFrame) -> av.VideoFrame:

    img_rgb = frame.to_ndarray(format="rgb24")

    if cam_image_load.check_button_pressed:
        cam_image_load.image = img_rgb
        cam_image_load.check_button_pressed = False

    return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")


with streaming_placeholder.container():
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

photo_doc = st.file_uploader("Upload the photo from the document")

if st.button('Confirm'):

    cam_image_load.check_button_pressed = True

    if photo_doc is not None:
        image_array = np.frombuffer(photo_doc.read(), dtype=np.uint8)
        image_doc = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image_doc = cv2.cvtColor(image_doc, cv2.COLOR_BGR2RGB)

        col_left, col_right = st.columns([1, 1])

        bbox_doc, shape_doc = get_shape(image_doc)
        bbox_cam, shape_cam = get_shape(cam_image_load.image)

        cam_image = cam_image_load.image

        doc_descriptor = face_recognition.compute_face_descriptor(image_doc, shape_doc)
        cam_descriptor = face_recognition.compute_face_descriptor(cam_image, shape_cam)

        doc_descriptor = np.array(list(doc_descriptor))
        cam_descriptor = np.array(list(cam_descriptor))

        a = np.linalg.norm(doc_descriptor - cam_descriptor)

        if a < 0.60:
            result = ':green[Identity has been successfully confirmed!]'
            bbox_color = (0, 255, 0)
        else:
            result = ':red[Identity has not been confirmed!]'
            bbox_color = (255, 0, 0)

        marked_face_doc = draw_facemask(image_doc, bbox_doc, shape_doc, bbox_color)
        marked_face_cam = draw_facemask(cam_image, bbox_cam, shape_cam, bbox_color)

        h_doc, w_doc = marked_face_doc.shape[:2]
        h_cam, w_cam = marked_face_cam.shape[:2]
        max_side_doc = max(h_doc, w_doc)
        max_side_cam = max(h_cam, w_cam)

        if max_side_cam > max_side_doc:
            scale = max_side_cam / max_side_doc
            marked_face_doc = cv2.resize(marked_face_doc, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        else:
            scale = max_side_doc / max_side_cam
            marked_face_cam = cv2.resize(marked_face_cam, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        col_left.image(marked_face_doc, use_column_width=False)
        col_right.image(marked_face_cam, use_column_width=False)

        st.subheader(result)
