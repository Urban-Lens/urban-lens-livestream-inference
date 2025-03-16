#!/usr/bin/env python3
"""
Frame Publisher

Publishes processed frames to Redis for the HLS Stream Worker.
This code would be integrated with your detection/processing pipeline.
"""

import os
import cv2
import json
import redis
import base64
import logging
import time
from datetime import datetime
import argparse
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('frame_publisher')

# Configuration (can be overridden by environment variables)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_CHANNEL = os.getenv('REDIS_CHANNEL', 'frame_updates')

def publish_frame(redis_client, source_id, frame, timestamp=None):
    """
    Publish a frame to Redis for streaming
    
    Args:
        redis_client: Redis client instance
        source_id: Stream source identifier
        frame: OpenCV image (numpy array)
        timestamp: Optional timestamp (defaults to current time)
    """
    try:
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Convert to base64 for JSON serialization
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        # Create message
        message = {
            'source_id': source_id,
            'timestamp': timestamp,
            'frame': encoded_frame
        }
        
        # Publish to Redis
        redis_client.publish(REDIS_CHANNEL, json.dumps(message))
        logger.debug(f"Published frame for {source_id}, timestamp: {timestamp}")
        
        return True
    except Exception as e:
        logger.error(f"Error publishing frame: {str(e)}")
        return False

def test_publish_sample_frames(source_id, num_frames=30, delay=1.0):
    """
    Test function to publish sample frames
    
    Args:
        source_id: Stream source identifier
        num_frames: Number of frames to publish
        delay: Delay between frames in seconds
    """
    # Connect to Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    
    # Create a sample image with text
    for i in range(num_frames):
        # Create a blank image
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add text with frame number and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Source: {source_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{num_frames}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {timestamp}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add a moving element
        x_pos = int(50 + (1180 * (i / num_frames)))
        cv2.circle(frame, (x_pos, 360), 50, (0, 0, 255), -1)
        
        # Publish frame
        publish_frame(redis_client, source_id, frame)
        
        logger.info(f"Published test frame {i+1}/{num_frames} for source {source_id}")
        
        # Wait before next frame
        time.sleep(delay)
    
    logger.info(f"Finished publishing {num_frames} test frames for {source_id}")

if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Frame Publisher Test")
    parser.add_argument("source_id", help="Source ID for the stream")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to publish")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between frames in seconds")
    parser.add_argument("--redis-host", help="Redis host", default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, help="Redis port", default=REDIS_PORT)
    parser.add_argument("--redis-channel", help="Redis channel", default=REDIS_CHANNEL)
    
    args = parser.parse_args()
    
    # Update config from args
    REDIS_HOST = args.redis_host
    REDIS_PORT = args.redis_port
    REDIS_CHANNEL = args.redis_channel
    
    # Run test with provided parameters
    test_publish_sample_frames(args.source_id, args.frames, args.delay)