import os
from ultralytics import YOLO, SAM
import cv2
from tqdm.notebook import tqdm
import supervision as sv

from supervision.draw.utils import draw_polygon


# load the model
def load_model(model_task, model_path: str = "/content/models/"):
    if model_task.lower() in ["detect", "detection"]:
        # /content/models/farm_detection.pt
        return YOLO(os.path.join(model_path, "farm_detection.pt"))

    if model_task.lower() in ["seg", "segment", "segmentation"]:
        # /content/models/sam_l.pt
        return SAM(os.path.join(model_path, "sam_l.pt"))

    return None


def load_image(image_path):
    image_bgr = cv2.imread(image_path)

    # convert to rgb
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# detect farms boundaries
def detect_farm(model, image, **kwargs):
    conf = kwargs["conf"]
    results = model(image, conf=conf)[0]

    return {"xyxy": results.boxes.xyxy.cpu().numpy()}


# segment farms with specified boundaries
def segment_farms(model, image, **kwargs):
    xyxy = kwargs["xyxy"]

    output = {"xyxy": [], "masks": []}

    for coord in tqdm(xyxy, desc="masking process"):
        results = model(image, bboxes=coord.reshape(1, -1))[0]
        output["xyxy"].append(coord.reshape(1, -1))
        output["masks"].append(results.masks.data.cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def get_detection(**kwargs):
    _ = kwargs["xyxy"]
    masks = kwargs["masks"]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)

    return detections


def plot_segment_farms(detections, image):
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red(), color_map="index")

    return mask_annotator.annotate(scene=image.copy(), detections=detections)


def plot_bbox_farms(detections, image):
    box_annotator = sv.BoxAnnotator(color=sv.Color.red())

    return box_annotator.annotate(
        scene=image.copy(), detections=detections, skip_label=True
    )


def plot_poly_farms(detections, image):
    polygons = [sv.detection.utils.mask_to_polygons(m) for m in detections.mask]

    image_output = image.copy()
    for poly in tqdm(polygons, desc="draw polygans"):
        image_output = sv.draw_polygon(
            scene=image_output, polygon=poly[0], color=sv.Color.red()
        )

    return image_output


IMAGE_PATH = "/content/data/farm-1.jpeg"
image = load_image(IMAGE_PATH)
sv.plot_image(image, size=(7, 5))


## Load Models Test

model_path = "/content/models/"
model_task = "detect"

# Load detection model
farm_detection_model = load_model(model_task=model_task, model_path=model_path)

# Load segment model
model_task = "seg"
farm_seg_model = load_model(model_task=model_task, model_path=model_path)
