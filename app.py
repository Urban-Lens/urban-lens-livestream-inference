from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import time
import os
from typing import Optional, Dict, List, Tuple
import uuid
import json
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult, ResultSet
import io
import time
import psycopg2
from psycopg2.extras import Json
import asyncio
from dotenv import load_dotenv
from api_postprocessing import save_to_s3

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Object Detection API", 
              description="API for detecting and counting objects in images using YOLO with Celery workers")

# Celery configuration
celery_app = Celery(
    "object_detection",
    broker=f"redis://{os.getenv('REDIS_HOST')}:6379/0",
    backend=f"redis://{os.getenv('REDIS_HOST')}:6379/0"
)

celery_app.conf.update(
    task_serializer='pickle',
    accept_content=['json', 'pickle'],
    result_serializer='pickle',
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

class DetectionResult(BaseModel):
    timestamp: float
    processing_time: float
    total_objects: int
    class_counts: Dict[str, int]
    image_dimensions: Dict[str, int]
    split_info: Dict[str, int]
    detections: List[BoundingBox]

def split_image(image: np.ndarray, n: int):
    """
    Split an image into n x n equal pieces.
    
    Args:
        image: NumPy array image (CV2 format)
        n: Number of splits in each dimension (resulting in n^2 pieces)
        
    Returns:
        List of tuples containing (image_piece, (x_offset, y_offset), piece_size)
    """
    height, width = image.shape[:2]
    piece_width = width // n
    piece_height = height // n
    
    pieces = []
    
    for i in range(n):
        for j in range(n):
            # Calculate coordinates for cropping
            left = j * piece_width
            upper = i * piece_height
            right = left + piece_width
            lower = upper + piece_height
            
            # Ensure the last pieces go to the edge of the image
            if i == n-1:
                lower = height
            if j == n-1:
                right = width
                
            # Crop the piece and store with its position
            piece = image[upper:lower, left:right]
            piece_size = (right - left, lower - upper)
            
            pieces.append((piece, (left, upper), piece_size))
            
    return pieces

def merge_detections(detection_results, image_shape):
    """
    Merge detections from multiple image pieces.
    
    Args:
        detection_results: List of (detections_dict, offset) tuples
        image_shape: (height, width) of the original image
    
    Returns:
        Merged detections in a format ready for the API response
    """
    merged_xyxy = []
    merged_confidence = []
    merged_class_id = []
    merged_boxes = []
    class_names = {}
    
    for detections_dict, (offset_x, offset_y) in detection_results:
        if not detections_dict or not detections_dict.get('xyxy'):
            continue
            
        # Get class names from the first successful detection
        if not class_names and 'class_names' in detections_dict:
            class_names = detections_dict['class_names']
            
        # Process each box
        for i, box in enumerate(detections_dict['xyxy']):
            # Adjust coordinates based on piece offset
            x1, y1, x2, y2 = box
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y
            
            # Add to merged lists
            merged_xyxy.append([x1, y1, x2, y2])
            
            if 'confidence' in detections_dict and i < len(detections_dict['confidence']):
                confidence = detections_dict['confidence'][i]
                merged_confidence.append(confidence)
            else:
                merged_confidence.append(0.0)
                
            if 'class_id' in detections_dict and i < len(detections_dict['class_id']):
                class_id = int(detections_dict['class_id'][i])
                merged_class_id.append(class_id)
            else:
                merged_class_id.append(0)
                
            # Create bounding box object
            class_id = merged_class_id[-1]
            class_name = class_names.get(class_id, "unknown")
            
            merged_boxes.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(merged_confidence[-1]),
                "class_id": class_id,
                "class_name": class_name
            })
    
    # Count classes
    class_counts = {}
    for class_id in merged_class_id:
        class_name = class_names.get(int(class_id), "unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        "boxes": merged_boxes,
        "class_counts": class_counts,
        "total_objects": len(merged_boxes)
    }

def log_time(label, start):
    elapsed = time.time() - start
    print(f"{label} took {elapsed:.4f} seconds")

# Database connection
def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    print(f"Using DATABASE_URL: {os.getenv('DATABASE_URL')}")
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def save_to_database(results, source_id):
    """Save detection results to the database"""
    try:
        conn = get_db_connection()
        if not conn:
            print("Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        
        # Count people and vehicles
        class_counts = results.get("class_counts", {})
        people_ct = class_counts.get("person", 0)
        vehicle_ct = class_counts.get("car", 0) + class_counts.get("truck", 0)
        output_img_path = results.get("output_img_path", "NULL")
        
        # Insert into the timeseries_analytics table
        query = """
            INSERT INTO timeseries_analytics
            (timestamp, source_id, people_ct, vehicle_ct, detections, output_img_path)
            VALUES (NOW(), %s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        cursor.execute(query, (
            source_id, 
            people_ct, 
            vehicle_ct, 
            Json(results),
            output_img_path
        ))
        
        record_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Saved detection results to database with ID: {record_id}")
        return True
        
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        if conn:
            conn.rollback()
            conn.close()
        return False

async def process_and_save_detections(
    image_data: bytes,
    source_id: str,
    split_n: int,
    confidence: float,
    timeout: int,
    celery_app: Celery
):
    """Process image with detection and save results to database and stream"""
    try:
        overall_start = time.time()
        
        # Decode image
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("Invalid image data")
            return
        
        height, width = image.shape[:2]
        
        # Split image if needed
        if split_n > 1:
            pieces = split_image(image, split_n)
        else:
            pieces = [(image, (0, 0), (width, height))]
        
        # Send to Celery workers
        tasks = []
        for piece, offset, _ in pieces:
            task = celery_app.send_task(
                'process_image_piece',
                args=[piece, offset, confidence]
            )
            tasks.append(task)
        
        # Wait for results
        task_results = []
        for task in tasks:
            try:
                result = task.get(timeout=timeout)
                task_results.append(result)
            except Exception as e:
                print(f"Task error or timeout: {str(e)}")
        
        # Merge detections
        merged_result = merge_detections(task_results, (height, width))
        
        processing_time = time.time() - overall_start
        
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
        
        # Get a copy of the image for processing
        processed_image = image.copy()
        
        # Draw bounding boxes and save image to S3
        s3_url, output_image = save_to_s3(
            processed_image, 
            merged_result["boxes"], 
            source_id, 
            merged_result["class_counts"]
        )

        # Add S3 URL to results if available
        if s3_url:
            results["output_img_path"] = s3_url

        # Save to database
        save_to_database(results, source_id)
        
        # Publish frame for streaming if Redis client is available
        try:
            # Import here to avoid circular imports
            from frame_publisher import publish_frame
            import redis
            
            # Connect to Redis (could be moved to a global connection)
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_channel = os.getenv('REDIS_CHANNEL', 'frame_updates')
            
            redis_client = redis.Redis(host=redis_host, port=redis_port)
            
            # Publish frame for streaming - use the same timestamp as results
            publish_result = publish_frame(
                redis_client,
                source_id, 
                output_image,  # Use the processed image with bounding boxes
                overall_start     # Use the same timestamp as in results
            )
            
            if publish_result:
                print(f"Published frame for streaming: {source_id}")
            
        except Exception as e:
            print(f"Error publishing frame to Redis: {str(e)}")
            # Continue execution - streaming is secondary to core functionality
        
        print(f"Processing complete for source_id {source_id} in {processing_time:.2f} seconds")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Background processing error: {str(e)}")

# Add this to your app:

@app.post("/save_detections/")
async def save_detections(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_id: str = Form(...),
    split_n: Optional[int] = Form(1),
    confidence: Optional[float] = Form(0.35),
    timeout: Optional[int] = Form(30)
):
    """
    Process image detection in the background and save results to database
    """
    try:
        # Read the file
        contents = await file.read()
        
        # Start background task
        background_tasks.add_task(
            process_and_save_detections,
            contents,
            source_id,
            split_n,
            confidence,
            timeout,
            celery_app
        )
        
        return {
            "status": "processing",
            "message": f"Image scheduled for processing with source_id: {source_id}",
            "source_id": source_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling processing: {str(e)}")

@app.post("/detect/", response_model=DetectionResult)
async def detect_objects(
    file: UploadFile = File(...),
    split_n: Optional[int] = Form(1),
    confidence: Optional[float] = Form(0.35),
    timeout: Optional[int] = Form(30)  # Timeout for waiting for worker results
):
    """
    Detect objects in the uploaded image using Celery workers.
    """
    overall_start = time.time()
    
    try:
        read_start = time.time()
        contents = await file.read()
        log_time("Reading file", read_start)
        
        decode_start = time.time()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        log_time("Decoding image", decode_start)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        height, width = image.shape[:2]
        
        split_start = time.time()
        if split_n > 1:
            pieces = split_image(image, split_n)
        else:
            pieces = [(image, (0, 0), (width, height))]
        log_time("Splitting image", split_start)
        
        task_start = time.time()
        tasks = []
        for piece, offset, _ in pieces:
            task = celery_app.send_task(
                'process_image_piece',
                args=[piece, offset, confidence]
            )
            tasks.append(task)
        log_time("Submitting tasks", task_start)
        
        result_start = time.time()
        task_results = []
        for task in tasks:
            try:
                result = task.get(timeout=timeout)
                task_results.append(result)
            except Exception as e:
                print(f"Task error or timeout: {str(e)}")
        log_time("Waiting for task results", result_start)
        
        merge_start = time.time()
        merged_result = merge_detections(task_results, (height, width))
        log_time("Merging detections", merge_start)
        
        processing_time = time.time() - overall_start
        
        response = {
            "timestamp": overall_start,
            "processing_time": processing_time,
            "total_objects": merged_result["total_objects"],
            "class_counts": merged_result["class_counts"],
            "image_dimensions": {"width": width, "height": height},
            "split_info": {"n": split_n, "pieces": split_n ** 2},
            "detections": merged_result["boxes"]
        }
        
        log_time("Overall processing", overall_start)
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Celery-based Object Detection API is running. Use /detect/ endpoint to analyze images."}

@app.get("/health")
def health_check():
    """
    Check if the API and Celery workers are healthy
    """
    try:
        # Try to ping the Celery workers
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if not stats:
            return {"status": "unhealthy", "message": "No Celery workers available"}
        
        worker_count = len(stats)
        
        return {
            "status": "healthy",
            "message": f"API running with {worker_count} Celery workers available",
            "workers": list(stats.keys())
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}

# Run the API with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)