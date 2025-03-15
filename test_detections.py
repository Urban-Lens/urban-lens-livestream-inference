#!/usr/bin/env python3
import cv2
import json
import sys
import requests
import numpy as np
import argparse

def detect_and_draw(image_path, api_url="http://localhost:8000/detect/", 
                   split_n=4, confidence=0.15, parallel=True):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Prepare image for API request
    _, img_encoded = cv2.imencode('.jpg', image)
    
    # Send request to API
    try:
        import time
        start_time = time.time()
        
        files = {"file": (image_path, img_encoded.tobytes(), "image/jpeg")}
        data = {
            "split_n": split_n,
            "confidence": confidence,
            "parallel": str(parallel).lower()
        }
        
        print(f"Sending request to {api_url}...")
        response = requests.post(api_url, files=files, data=data)
        
        total_time = time.time() - start_time
        print(f"Total request time: {total_time:.3f} seconds")
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return
        
        results = response.json()
    except Exception as e:
        print(f"Error sending request: {str(e)}")
        return
    
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
    
    # Draw bounding boxes
    detections = results.get('detections', [])
    for det in detections:
        x1, y1 = int(det['x1']), int(det['y1'])
        x2, y2 = int(det['x2']), int(det['y2'])
        class_name = det['class_name']
        confidence = det['confidence']
        
        color = colors.get(class_name, default_color)
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1-label_size[1]-5), (x1+label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add summary at top
    class_counts = results.get('class_counts', {})
    total_objects = results.get('total_objects', 0)
    proc_time = results.get('processing_time', 0)
    
    # Background for summary
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    
    # Total count and processing time
    cv2.putText(image, f"Total: {total_objects} | Processing: {proc_time:.3f}s", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Class counts
    count_text = " | ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
    cv2.putText(image, count_text, (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the image
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects and display results")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("-a", "--api", default="http://localhost:8000/detect/", help="API URL")
    parser.add_argument("-s", "--split", type=int, default=4, help="Split number")
    parser.add_argument("-c", "--confidence", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("-p", "--parallel", type=lambda x: x.lower() == 'true', 
                        default=True, help="Use parallel processing (true/false)")
    
    args = parser.parse_args()
    
    detect_and_draw(
        args.image_path, 
        args.api, 
        args.split, 
        args.confidence, 
        args.parallel
    )