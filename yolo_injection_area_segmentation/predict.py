from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("./yolo11n-seg.pt")  # load a custom model
model = YOLO("runs/segment/train6/weights/best.pt")

# Predict with the model
results = model([
    "/home/yan/StudioSpace/AI_Annotation_Studio/core_work_flow/storage/ComfyUI/output/flux_00376_.png",])  # predict on an image

# Access the results
# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)



for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk