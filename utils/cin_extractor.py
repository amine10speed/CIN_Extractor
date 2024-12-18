import os
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from PIL import Image
import label_map_util
import visualization_utils as vis_util
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper

# Paths to the object detection model and label map
MODEL_NAME = 'model'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('data', 'labelmap.pbtxt')
NUM_CLASSES = 9

# Hardcoded field positions for the cropped CIN (800x445)
field_positions = {
    "name": (289, 89, 161, 54),
    "family_name": (288, 145, 167 , 30),
    "date_of_birth": (460, 163, 139 , 38),
    "place_of_birth": (295, 207, 338 , 47),
    "id_number": (88, 375, 151  , 68 ),
    "expiration_date": (561, 376, 115 , 69),
    "arabic_name": (600, 71, 200 , 45 ),
    "arabic_family_name": (616, 121, 184 , 43 ),
    "place_of_birth_arabic": (584, 197, 187 , 44 ),
}

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors for the object detection classifier
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Function to apply OCR on cropped fields
def extract_text_from_field(image, position, field_name, debug=False, is_arabic=False):
    x, y, w, h = position
    cropped_region = image[y:y+h, x:x+w]

    # Save the cropped region for debugging
    if debug:
        debug_path = f"debug_{field_name}.png"
        cv2.imwrite(debug_path, cropped_region)

    # Preprocess the cropped region
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2GRAY)

    if is_arabic:
        # Dialed-back preprocessing for Arabic fields
        gray = cv2.GaussianBlur(gray, (1, 1), 0)  # Reduce GaussianBlur kernel size
        _, binary = cv2.threshold(gray, 155, 250, cv2.THRESH_BINARY)  # Adjust threshold value
    else:
        # Use slightly adjusted binarization for other fields
        _, binary = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)

    # Save the preprocessed binary image for debugging
    if debug:
        debug_binary_path = f"debug_binary_{field_name}.png"
        cv2.imwrite(debug_binary_path, binary)

    # Apply OCR
    lang = 'ara' if is_arabic else 'eng'
    text = pytesseract.image_to_string(binary, lang=lang).strip()

    # Reorder and reshape Arabic text
    if is_arabic and text:
        text = get_display(arabic_reshaper.reshape(text))

    return text



# Detect CIN card, crop it, and find fields
def detect_and_process_cin(image_path, output_image_path, min_score_thresh=0.6):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    image_height, image_width, _ = image.shape

    # Run object detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Identify the largest bounding box
    largest_box = None
    max_area = 0
    for i in range(int(num[0])):
        if scores[0][i] > min_score_thresh:
            box = boxes[0][i]
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (int(xmin * image_width), int(xmax * image_width),
                                         int(ymin * image_height), int(ymax * image_height))
            area = (right - left) * (bottom - top)
            if area > max_area:
                max_area = area
                largest_box = (left, right, top, bottom)

    # Crop and resize the detected CIN card
    cropped_resized_path = None
    extracted_fields = {}
    if largest_box:
        left, right, top, bottom = largest_box
        cropped_cin = image_rgb[top:bottom, left:right]
        resized_cin = cv2.resize(cropped_cin, (800, 445))
        cropped_resized_path = 'cropped_resized_cin.jpg'
        cv2.imwrite(cropped_resized_path, cv2.cvtColor(resized_cin, cv2.COLOR_RGB2BGR))

        # Extract fields using predefined positions
        for field_name, position in field_positions.items():
            is_arabic = "arabic" in field_name
            extracted_fields[field_name] = extract_text_from_field(resized_cin, position, field_name, debug=True, is_arabic=is_arabic)

    # Visualize detection boxes on original image
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=min_score_thresh)
    cv2.imwrite(output_image_path, image)

    return extracted_fields, output_image_path, cropped_resized_path

# Main function to process the CIN card
def process_cin_card(image_path):
    output_image_path = 'output_detected_image.jpg'
    try:
        # Extract data using detect_and_process_cin
        extracted_fields, annotated_image, cropped_resized_path = detect_and_process_cin(image_path, output_image_path)

        # Return the extracted data and file paths
        return extracted_fields, annotated_image, cropped_resized_path
    except Exception as e:
        # Handle errors gracefully and log them if necessary
        print(f"Error processing CIN card: {e}")
        return None



