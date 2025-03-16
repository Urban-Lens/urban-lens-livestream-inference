#!/usr/bin/env python3
"""
Location Streams Monitor

This script:
1. Reads location data from the database
2. For each location with an input_stream_url, starts a separate process to monitor the stream
3. Uses the location ID as the source ID for frame capturing

Prerequisites:
- PostgreSQL database connection
- Running detection API
"""

import os
import psycopg2
import psycopg2.extras
import argparse
import subprocess
import time
import signal
import sys
from dotenv import load_dotenv
import logging
import streamlink
import cv2
from datetime import datetime
import requests


# Load environment variables
load_dotenv()

# Database connection string
DATABASE_URL = os.getenv('DATABASE_URL')

# Other settings (can be overridden with command-line args)
API_URL = os.getenv('API_URL', 'http://localhost:8000')
FRAME_CAPTURE_SCRIPT = os.getenv('FRAME_CAPTURE_SCRIPT', 'stream_detector.py')
SPLIT_N = int(os.getenv('SPLIT_N', '4'))
CONFIDENCE = float(os.getenv('CONFIDENCE', '0.15'))
PARALLEL = os.getenv('PARALLEL', 'true').lower() in ('true', 'yes', '1')

# Process tracking
process_map = {}

def get_db_connection():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def get_active_locations():
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT id, input_stream_url
            FROM location
            WHERE input_stream_url IS NOT NULL AND input_stream_url != ''
        """
        cursor.execute(query)
        locations = cursor.fetchall()
        cursor.close()
        conn.close()
        return locations
    except Exception as e:
        print(f"Error fetching locations: {e}")
        return []

def start_stream_process(location, api_url, split_n, confidence, parallel):
    """Start a stream monitoring process for a location"""
    location_id = str(location['id'])
    stream_url = location['input_stream_url']
    
    # Prepare command
    cmd = [
        'python', FRAME_CAPTURE_SCRIPT,
        stream_url,
        '--api', api_url,
        '--split', str(split_n),
        '--confidence', str(confidence),
        '--parallel', str(parallel).lower(),
        '--source-id', location_id
    ]
    
    # Start process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            preexec_fn=os.setsid  # Used to terminate the whole process group later
        )
        
        print(f"Started monitoring for location {location_id} ({location['address']})")
        print(f"Stream URL: {stream_url}")
        print(f"Process PID: {process.pid}")
        print("-" * 50)
        
        return process
    except Exception as e:
        print(f"Error starting process for location {location_id}: {str(e)}")
        return None

def stop_all_processes():
    """Stop all running stream processes"""
    for location_id, process in process_map.items():
        try:
            if process and process.poll() is None:  # If process is still running
                # Send the signal to the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                print(f"Stopped monitoring for location {location_id}")
        except Exception as e:
            print(f"Error stopping process for location {location_id}: {str(e)}")

def handle_exit(sig, frame):
    """Handle exit signals gracefully"""
    print("\nShutting down...")
    stop_all_processes()
    sys.exit(0)

def send_frame_to_api(frame, api_url, source_id, split_n, confidence, parallel):
    """Send frame to the save_detections API endpoint"""
    try:
        # Convert frame to jpg format
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # Prepare the request
        files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {
            "source_id": source_id,
            "split_n": split_n,
            "confidence": confidence,
            "parallel": str(parallel).lower()
        }
        
        # Send request
        api_url = api_url.rstrip('/') + '/save_detections/'
        response = requests.post(api_url, files=files, data=data)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

def process_stream(youtube_url, cookies_file, api_url, split_n, confidence, parallel, source_id):
    """Process a stream continuously, capturing frames at regular intervals"""
    # Set up thread-specific logger
    thread_logger = logging.getLogger(f"stream_{source_id}")
    thread_logger.setLevel(logging.INFO)
    
    # Add console handler if not already added
    if not thread_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'[%(asctime)s] [%(name)s] %(message)s')
        handler.setFormatter(formatter)
        thread_logger.addHandler(handler)
    
    # Set up Streamlink session
    session = streamlink.Streamlink()
    
    # Configure session (could add options here if needed)
    session.set_option("stream-timeout", 30)
    session.set_option("stream-segment-timeout", 10)
    
    # Normalize API URL
    if api_url.endswith('/save_detections/'):
        api_url = api_url.replace('/save_detections/', '/')
    
    thread_logger.info(f"Starting stream processing for {source_id}: {youtube_url}")
    thread_logger.info(f"API URL: {api_url}")
    
    frame_count = 0
    connection_errors = 0
    frame_interval = 5  # Seconds between frames
    max_errors = 10  # Maximum consecutive errors before backing off
    
    try:
        while True:
            try:
                # Get stream info (with backoff on repeated errors)
                if connection_errors > max_errors:
                    sleep_time = min(30, connection_errors)
                    thread_logger.warning(f"Too many errors ({connection_errors}), backing off for {sleep_time}s")
                    time.sleep(sleep_time)
                
                # Get a fresh stream URL for every capture to ensure latest content
                streams = session.streams(youtube_url)
                if not streams:
                    thread_logger.warning("No streams found, retrying in 5s")
                    connection_errors += 1
                    time.sleep(5)
                    continue
                    
                # Reset error counter on success
                connection_errors = 0
                
                # Get the best quality stream
                stream_url = streams["best"].url
                
                # Open a new capture for each frame
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    thread_logger.warning("Could not open stream, retrying in 5s")
                    time.sleep(5)
                    continue
                
                # Get a single frame
                ret, frame = cap.read()
                
                # Close the capture immediately
                cap.release()
                
                if not ret:
                    thread_logger.warning("Failed to get frame, retrying in 5s")
                    time.sleep(5)
                    continue
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Send frame to API
                thread_logger.info(f"Sending frame {frame_count + 1} to API")
                api_start_time = time.time()
                result = send_frame_to_api(frame, api_url, source_id, split_n, confidence, parallel)
                api_time = time.time() - api_start_time
                
                if result:
                    thread_logger.info(f"Frame scheduled for processing (took {api_time:.3f}s)")
                    frame_count += 1
                else:
                    thread_logger.warning("Failed to send frame to API")
                
                # Wait until next frame time
                time.sleep(frame_interval)
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                thread_logger.error(f"Error processing frame: {str(e)}")
                connection_errors += 1
                time.sleep(5)  # Wait before retry
                
    except KeyboardInterrupt:
        thread_logger.info(f"Stream processing interrupted after {frame_count} frames")
    except Exception as e:
        thread_logger.error(f"Fatal error in stream processing: {str(e)}")
    
    thread_logger.info(f"Stream processing for {source_id} ending after {frame_count} frames")
    return frame_count

# Integration into your existing script
if __name__ == "__main__":
    # Get locations from database
    locations = get_active_locations()
    
    # Process each location in parallel
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for location in locations:
            source_id = location['id']
            stream_url = location['input_stream_url']
            
            futures.append(
                executor.submit(
                    process_stream,
                    stream_url,
                    None,  # cookies_file
                    API_URL,
                    SPLIT_N,
                    CONFIDENCE,
                    PARALLEL,
                    source_id
                )
            )
        
        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing stream: {e}")