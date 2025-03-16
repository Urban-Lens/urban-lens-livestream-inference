#!/usr/bin/env python3
"""
Location Streams Monitor

This script:
1. Reads location data from the database
2. For each location with an input_stream_url, starts a thread to monitor the stream
3. Uses the location ID as the source ID for frame capturing
4. Periodically checks for new locations and updates accordingly

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
import threading
import concurrent.futures
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('frame_extractor')

# Load environment variables
load_dotenv()

# Database connection string
DATABASE_URL = os.getenv('DATABASE_URL')

# Other settings (can be overridden with command-line args)
API_URL = os.getenv('API_URL', 'http://localhost:8000')
SPLIT_N = int(os.getenv('SPLIT_N', '4'))
CONFIDENCE = float(os.getenv('CONFIDENCE', '0.15'))
PARALLEL = os.getenv('PARALLEL', 'true').lower() in ('true', 'yes', '1')
LOCATION_CHECK_INTERVAL = int(os.getenv('LOCATION_CHECK_INTERVAL', '300'))  # Seconds (5 minutes by default)
FRAME_INTERVAL = float(os.getenv('FRAME_INTERVAL', '5.0'))  # Seconds between frames

# Active threads and futures tracking
active_locations = {}  # Track currently monitored locations
executor = None  # Will be initialized as a ThreadPoolExecutor
location_lock = threading.Lock()  # Lock for thread-safe updates to active_locations

def get_db_connection():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return None

def get_active_locations():
    """Fetch locations with active streams from the database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT id, input_stream_url, address
            FROM location
            WHERE input_stream_url IS NOT NULL AND input_stream_url != ''
        """
        cursor.execute(query)
        locations = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return locations
    except Exception as e:
        logger.error(f"Error fetching locations: {e}")
        if conn:
            conn.close()
        return []

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
            logger.error(f"Error: API returned status code {response.status_code}")
            logger.debug(f"Response: {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        logger.error(f"Error sending request: {e}")
        return None

def process_stream(youtube_url, api_url, split_n, confidence, parallel, source_id, address=None):
    """Process a stream continuously, capturing frames at regular intervals"""
    # Set up thread-specific logger
    thread_logger = logging.getLogger(f"stream_{source_id}_{address}")
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
    max_errors = 10  # Maximum consecutive errors before backing off
    
    try:
        while source_id in active_locations:  # Check if this location is still active
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
                
                # Wait until next frame time - check more frequently if this location is removed
                for _ in range(int(FRAME_INTERVAL * 2)):  # Check twice per interval
                    if source_id not in active_locations:
                        break
                    time.sleep(0.5)
                    
            except Exception as e:
                thread_logger.error(f"Error processing frame: {str(e)}")
                connection_errors += 1
                time.sleep(5)  # Wait before retry
                
    except Exception as e:
        thread_logger.error(f"Fatal error in stream processing: {str(e)}")
    
    thread_logger.info(f"Stream processing for {source_id} ending after {frame_count} frames")
    return frame_count

def location_monitor():
    """
    Background thread that periodically checks for new or removed locations
    without impacting frame capture timing
    """
    logger.info(f"Starting location monitor thread (check interval: {LOCATION_CHECK_INTERVAL}s)")
    
    while True:
        try:
            # Fetch current active locations from the database
            db_locations = get_active_locations()
            db_location_ids = {str(loc['id']) for loc in db_locations}
            
            with location_lock:
                # Find new locations to add
                current_location_ids = set(active_locations.keys())
                new_location_ids = db_location_ids - current_location_ids
                removed_location_ids = current_location_ids - db_location_ids
                
                # Start monitoring new locations
                for loc in db_locations:
                    loc_id = str(loc['id'])
                    if loc_id in new_location_ids:
                        address = loc['address']
                        logger.info(f"Adding new location: {loc_id} - {address}")
                        stream_url = loc['input_stream_url']
                        
                        # Submit task to process this stream
                        future = executor.submit(
                            process_stream,
                            stream_url,
                            API_URL,
                            SPLIT_N,
                            CONFIDENCE,
                            PARALLEL,
                            loc_id,
                            address
                        )
                        
                        # Store the future and stream URL
                        active_locations[loc_id] = {
                            'future': future,
                            'url': stream_url,
                            'address': address
                        }
                
                # Remove locations that no longer exist
                for loc_id in removed_location_ids:
                    logger.info(f"Removing location: {loc_id}")
                    # The thread will exit on its own after checking active_locations
                    del active_locations[loc_id]
            
            # Log status
            logger.info(f"Currently monitoring {len(active_locations)} locations")
            
        except Exception as e:
            logger.error(f"Error in location monitor: {str(e)}")
            
        # Sleep until next check
        time.sleep(LOCATION_CHECK_INTERVAL)

def handle_exit(sig, frame):
    """Handle exit signals gracefully"""
    logger.info("Shutting down...")
    
    # Stop all stream processing
    with location_lock:
        active_locations.clear()  # This will signal threads to stop
    
    # Shutdown executor
    if executor:
        executor.shutdown(wait=False)
    
    logger.info("All streams stopped")
    sys.exit(0)

def main():
    """Main function"""
    global executor
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    logger.info("Starting frame extractor...")
    logger.info(f"API URL: {API_URL}")
    logger.info(f"Database URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configured'}")
    logger.info(f"Checking for new locations every {LOCATION_CHECK_INTERVAL} seconds")
    logger.info(f"Frame interval: {FRAME_INTERVAL} seconds")
    logger.info("=" * 50)
    
    # Create thread pool
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)
    
    try:
        # Start location monitor in background thread
        monitor_thread = threading.Thread(target=location_monitor, daemon=True)
        monitor_thread.start()
        
        # Wait indefinitely (the monitor thread will handle everything)
        monitor_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        handle_exit(None, None)
    except Exception as e:
        logger.error(f"Error in main thread: {str(e)}")
        handle_exit(None, None)

if __name__ == "__main__":
    main()