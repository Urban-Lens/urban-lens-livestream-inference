# worker.py
from celery import Celery
import os
import time
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

# Configure Celery
celery_app = Celery(
    "object_detection",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

celery_app.conf.update(
    task_serializer='pickle',
    accept_content=['json', 'pickle'],
    result_serializer='pickle',
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Model path - looks in the current working directory on the worker
model_path = os.environ.get("YOLO_MODEL_PATH", "yolo11n.onnx")
# Load model (on worker)
model = YOLO(model_path)

# Check if YOLO model exists
model_path = os.environ.get("YOLO_MODEL_PATH", "yolo11n.onnx")
if os.path.exists(model_path):
    print(f"YOLO model found at: {model_path}")
else:
    print(f"WARNING: YOLO model not found at: {model_path}")

@celery_app.task(name="process_image_piece", serializer='pickle')
def process_image_piece(image_piece, offset, confidence=0.35):
    """
    Celery task to process a single image piece with YOLO.
    
    Args:
        image_piece: Numpy array of the image piece
        offset: (x, y) offset of the piece in the original image
        confidence: Confidence threshold
        
    Returns:
        Tuple of (detections_dict, offset) where detections_dict 
        contains serializable detection data
    """ 

    try:
        # Validate that the model path is not a directory
        # if os.path.isdir(model_path):
        #     raise IsADirectoryError(f"Model path is a directory: {model_path}")
                
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
                
        
        
        # Original size
        original_height, original_width = image_piece.shape[:2]
        
        # # Set target size for resizing if needed
        # target_size = (640, 480)
        
        # # Resize if needed
        # if original_width < target_size[0] or original_height < target_size[1]:
        #     # Calculate ratios
        #     width_ratio = target_size[0] / original_width
        #     height_ratio = target_size[1] / original_height
            
        #     # Use the smaller ratio to preserve aspect ratio
        #     resize_ratio = min(width_ratio, height_ratio)
            
        #     # Calculate new dimensions
        #     new_width = int(original_width * resize_ratio)
        #     new_height = int(original_height * resize_ratio)
            
        #     # Resize the image
        #     resized_piece = cv2.resize(image_piece, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
        #     # Calculate scaling factors
        #     scale_x = original_width / new_width
        #     scale_y = original_height / new_height
        # else:
        #     # No resizing needed
        resized_piece = image_piece
        scale_x, scale_y = 1.0, 1.0
        
        # Run YOLO on the image piece
        result = model.predict(resized_piece, conf=confidence)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Prepare detections for serialization
        detections_dict = {}
        
        # Convert supervision Detections to serializable format
        if len(detections.xyxy) > 0:
            # Scale detections back to original size if resized
            scaled_xyxy = detections.xyxy.copy()
            scaled_xyxy[:, 0] *= scale_x  # x1
            scaled_xyxy[:, 2] *= scale_x  # x2
            scaled_xyxy[:, 1] *= scale_y  # y1
            scaled_xyxy[:, 3] *= scale_y  # y3
            
            # Prepare serializable dict
            detections_dict = {
                'xyxy': scaled_xyxy.tolist(),
                'confidence': detections.confidence.tolist() if detections.confidence is not None else [],
                'class_id': detections.class_id.tolist() if detections.class_id is not None else [],
                'class_names': {int(idx): name for idx, name in model.names.items()}
            }
        
        return (detections_dict, offset)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing image piece: {str(e)}")
        # Return empty detections on error
        return ({"xyxy": [], "confidence": [], "class_id": [], "class_names": {}}, offset)

if __name__ == "__main__":
    # Worker startup logic - print some diagnostic info
    print("Starting Celery worker for object detection...")
        
    # Print user info to confirm we're not running as root anymore
    import pwd, os
    user_info = pwd.getpwuid(os.getuid())
    print(f"Running as user: {user_info.pw_name} (uid={user_info.pw_uid})")
        
    # Start the worker
    celery_app.worker_main(["worker", "--loglevel=info", "-c", "1", "--without-heartbeat"])