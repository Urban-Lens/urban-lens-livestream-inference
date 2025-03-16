import os
import boto3
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

def save_to_s3(image, detections, source_id, class_counts=None):
    """
    Draw bounding boxes on the image and save to S3
    
    Args:
        image: numpy array (OpenCV format)
        detections: List of detection dictionaries with bounding box info
        source_id: ID of the source stream/location
        class_counts: Optional dictionary of class counts to display
    
    Returns:
        The S3 URL of the saved image
    """
    try:
        # Check if S3 configuration is available
        s3_output_path = "detections"
        s3_bucket = os.getenv('S3_BUCKET_NAME')
        
        # Create a copy of the image to draw on
        image_with_boxes = image.copy()
        
        # Define colors for different classes (BGR format)
        colors = {
            'person': (0, 255, 0),     # Green
            'car': (0, 0, 255),        # Red
            'truck': (255, 0, 0),      # Blue
            'bird': (255, 255, 0),     # Cyan
            'traffic light': (255, 0, 255),  # Magenta
            'potted plant': (0, 255, 255),   # Yellow
        }
        default_color = (255, 255, 255)  # White

        allowed_classes = ("person", "car", "bus", "truck")
        
        # Draw bounding boxes
        for det in detections:
            x1, y1 = int(det['x1']), int(det['y1'])
            x2, y2 = int(det['x2']), int(det['y2'])
            class_name = det['class_name']
            confidence = det['confidence']
            if class_name in allowed_classes:
                
                color = colors.get(class_name, default_color)
                
                # Draw box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image_with_boxes, 
                            (x1, y1-label_size[1]-5), 
                            (x1+label_size[0], y1), 
                            color, -1)
                cv2.putText(image_with_boxes, 
                        label, 
                        (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0), 
                        1)
        
        # Add summary at top
        if class_counts:
            total_objects = sum(class_counts.values())
            
            # Background for summary
            cv2.rectangle(image_with_boxes, (0, 0), (image_with_boxes.shape[1], 30), (0, 0, 0), -1)
            
            # Total count
            cv2.putText(image_with_boxes, 
                       f"Total: {total_objects}", 
                       (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (255, 255, 255), 
                       1)
            
            # Class counts
            count_text = " | ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            cv2.putText(image_with_boxes, 
                       count_text, 
                       (150, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (255, 255, 255), 
                       1)
        
        # Add source ID and timestamp at bottom
        height, width = image_with_boxes.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(image_with_boxes, 
                   f"Source: {source_id} | Time: {timestamp}", 
                   (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (255, 255, 255), 
                   1)
        
        # Resize image to max 1280x720 using thumbnail function
        # Convert OpenCV image to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        
        # Set maximum size
        max_size = (800, 800)
        
        # Use thumbnail method to resize (always maintains aspect ratio)
        image_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert back to OpenCV format
        image_with_boxes = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Create timestamp for filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_id}_detection_{timestamp_str}.png"
        
        # Convert image to bytes
        _, buffer = cv2.imencode(".png", image_with_boxes)
        image_bytes = buffer.tobytes()
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3'
        )
        
        # Upload to S3
        s3_key = f"{s3_output_path.rstrip('/')}/{filename}"
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=image_bytes,
            ContentType='image/png'
        )
        
        # Generate the URL for the uploaded image
        s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        
        print(f"Saved detection image to S3: {s3_url}")
        return s3_url
        
    except Exception as e:
        print(f"Error saving image to S3: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example of adding this function to the background processing flow:
'''
# In your process_and_save_detections function, add:

# Prepare final results
results = {
    "timestamp": overall_start,
    "processing_time": processing_time,
    "total_objects": merged_result["total_objects"],
    "class_counts": merged_result["class_counts"],
    "image_dimensions": {"width": width, "height": height},
    "split_info": {"n": split_n, "pieces": split_n ** 2},
    "detections": merged_result["boxes"]
}

# Draw bounding boxes and save image to S3
s3_url = save_to_s3(
    image, 
    merged_result["boxes"], 
    source_id, 
    merged_result["class_counts"]
)

# Add S3 URL to results if available
if s3_url:
    results["output_img_path"] = s3_url

# Save to database
save_to_database(results, source_id)
'''
