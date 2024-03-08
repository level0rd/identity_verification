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
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
THRESHOLD = 0.60
RECTANGLE_THICKNESS = 2
FACE_THICKNESS = 1

# Models loading
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat')
face_recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

streaming_placeholder = st.empty()


class CameraImageLoad:
    def __init__(self):
        self.check_button_pressed = False
        self.image_cam = None


camera_image_load = CameraImageLoad()


def shape_to_np(shape: dlib.full_object_detection, dtype: str = "int") -> np.ndarray:
    """
    Convert shape to a numpy array.

    Args:
        shape: The shape object to be converted.
        dtype (str): Data type of the numpy array.

    Returns:
        np.ndarray: Converted numpy array.
    """

    cord = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        cord[i] = (shape.part(i).x, shape.part(i).y)
    return cord


def get_descriptor(image: np.ndarray) -> tuple:
    """
    Compute face descriptor.

    Args:
        image: Image data.

    Returns:
        tuple: Bounding box and face descriptor.
    """

    bbox, shape = get_shape(image)
    return bbox, face_recognition.compute_face_descriptor(image, shape)


def get_shape(image: np.ndarray) -> tuple:
    """
    Obtain key points and bounding boxes.

    Args:
        image: Image data.

    Returns:
        tuple: Bounding box and shape object.
    """

    dets = detector(image, 1)

    for face_index, face_rectangle in enumerate(dets):
        shape = shape_predictor(image, face_rectangle)

        numbers = re.findall(r'\d+', str(face_rectangle))
        temp_str = ', '.join(numbers)
        bbox = temp_str.split(',')

    return bbox, shape


def draw_facemask(face: np.ndarray, bbox: list, shape: dlib.full_object_detection, bbox_color: tuple) -> np.ndarray: 
    """
    Draw facemask on the face.

    Args:
        face: Input face image.
        bbox: Bounding box coordinates.
        shape: Shape object representing face landmarks.
        bbox_color: Color for bounding box.

    Returns:
        np.ndarray: Modified face image with facemask.
    """

    points = shape_to_np(shape)

    cv2.(face, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), bbox_color, RECTANGLE_THICKNESS)

    # Lower border of the face
    for i in range(16):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)

    # Lips
    for i in range(48, 60):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)

    # Nose
    for i in range(27, 35):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)
    cv2.line(face, points[35], points[30], BLUE, FACE_THICKNESS)

    # Right eye
    for i in range(36, 41):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)
    cv2.line(face, points[36], points[41], BLUE, FACE_THICKNESS)

    # Left eye
    for i in range(42, 47):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)
    cv2.line(face, points[42], points[47], BLUE, FACE_THICKNESS)

    # Right brow
    for i in range(17, 21):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)

    # Left brow
    for i in range(22, 26):
        cv2.line(face, points[i], points[i + 1], BLUE, FACE_THICKNESS)

    x1 = max(0, int(bbox[0]) - PADDING)
    y1 = max(0, int(bbox[1]) - PADDING)
    x2 = min(face.shape[1], int(bbox[2]) + PADDING)
    y2 = min(face.shape[0], int(bbox[3]) + PADDING)

    return face[y1:y2, x1:x2]


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Callback function for processing video frames.

    Args:
        frame: Input video frame.

    Returns:
        av.VideoFrame: Processed video frame.
    """
    image_rgb = frame.to_ndarray(format="rgb24")

    if camera_image_load.check_button_pressed:
        camera_image_load.image = image_rgb
        camera_image_load.check_button_pressed = False

    return av.VideoFrame.from_ndarray(image_rgb, format="rgb24")


with streaming_placeholder.container():
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

photo_document = st.file_uploader("Upload the photo from the document")

if st.button('Confirm'):

    camera_image_load.check_button_pressed = True

    if photo_document is not None:
        image_array = np.frombuffer(photo_document.read(), dtype=np.uint8)
        image_document = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image_document = cv2.cvtColor(image_document, cv2.COLOR_BGR2RGB)

        left_column, right_column = st.columns([1, 1])

        bbox_document, shape_document = get_shape(image_document)
        bbox_camera, shape_camera = get_shape(camera_image_load.image)

        image_camera = camera_image_load.image

        document_descriptor = face_recognition.compute_face_descriptor(image_document, shape_document)
        camera_descriptor = face_recognition.compute_face_descriptor(image_camera, shape_camera)

        document_descriptor = np.array(list(document_descriptor))
        camera_descriptor = np.array(list(camera_descriptor))

        a = np.linalg.norm(document_descriptor - camera_descriptor)

        if a < THRESHOLD:
            result = ':green[Identity has been successfully confirmed!]'
            bbox_color = GREEN
        else:
            result = ':red[Identity has not been confirmed!]'
            bbox_color = RED

        marked_face_document = draw_facemask(image_document, bbox_document, shape_document, bbox_color)
        marked_face_camera = draw_facemask(image_camera, bbox_camera, shape_camera, bbox_color)

        height_document, width_document = marked_face_document.shape[:2]
        height_camera, width_camera = marked_face_camera.shape[:2]
        max_side_document = max(height_document, width_document)
        max_side_camera = max(height_camera, width_camera)

        if max_side_camera > max_side_document:
            scale = max_side_camera / max_side_document
            marked_face_document = cv2.resize(marked_face_document, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        else:
            scale = max_side_document / max_side_camera
            marked_face_camera = cv2.resize(marked_face_camera, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        left_column.image(marked_face_document, use_column_width=False)
        right_column.image(marked_face_camera, use_column_width=False)

        st.subheader(result)
