#!/usr/bin/env python3
import cv2
import streamlink
import json
import requests
import time
import argparse
from datetime import datetime

def parse_cookies_file(cookies_file):
    """Parse a Netscape/Mozilla cookies.txt file into a dictionary"""
    cookies = {}
    
    try:
        with open(cookies_file, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or line.strip() == '':
                    continue
                    
                # Split the line by tabs
                fields = line.strip().split('\t')
                
                # Check if line has enough fields
                if len(fields) >= 7:
                    domain, domain_specified, path, secure, expires, name, value = fields[:7]
                    
                    # Add to cookies dictionary
                    cookies[name] = value
    except Exception as e:
        print(f"Error parsing cookies file: {e}")
                    
    return cookies

def detect_frame(frame, api_url, split_n, confidence, parallel):
    """Send frame to API and get detections"""
    try:
        # Convert frame to jpg format
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # Prepare the request
        files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {
            "split_n": split_n,
            "confidence": confidence,
            "parallel": str(parallel).lower()
        }
        
        # Send request
        response = requests.post(api_url, files=files, data=data)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return None
            
        return response.json()
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

def process_stream(youtube_url, cookies_file, api_url, split_n, confidence, parallel, source_id):
    """Process stream for one minute, getting latest frame each second"""
    # Parse cookies
    cookies = parse_cookies_file(cookies_file)
    
    # Set up Streamlink session
    session = streamlink.Streamlink()
    
    # Configure cookies if available
    if cookies:
        session.set_option("http-cookies", cookies)
    
    # Start time
    start_time = time.time()
    end_time = start_time + 60  # Run for 1 minute
    
    print(f"Processing stream: {youtube_url}")
    print(f"Source ID: {source_id}")
    print(f"API URL: {api_url}")
    print(f"Will run for 60 seconds, getting one frame per second...")
    
    frame_count = 0
    
    while time.time() < end_time:
        try:
            # Get a fresh stream URL for every capture to ensure latest content
            streams = session.streams(youtube_url)
            if not streams:
                print("No streams found")
                time.sleep(1)
                continue
                
            stream_url = streams["best"].url
            
            # Open a new capture for each frame to get latest content
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("Could not open stream")
                time.sleep(1)
                continue
            
            # Get a single frame
            ret, frame = cap.read()
            
            # Close the capture immediately
            cap.release()
            
            if not ret:
                print("Failed to get frame")
                time.sleep(1)
                continue
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Send frame to API
            print(f"[{timestamp}] Sending frame {frame_count + 1} to API...")
            api_start_time = time.time()
            results = detect_frame(frame, api_url, split_n, confidence, parallel)
            api_time = time.time() - api_start_time
            
            if results:
                # Extract people and vehicle counts
                class_counts = results.get('class_counts', {})
                people_ct = class_counts.get('person', 0)
                vehicle_ct = class_counts.get('car', 0) + class_counts.get('truck', 0)
                total_objects = results.get('total_objects', 0)
                proc_time = results.get('processing_time', 0)
                
                print(f"[{timestamp}] Results: {total_objects} objects | People: {people_ct} | Vehicles: {vehicle_ct} | API time: {api_time:.3f}s | Processing: {proc_time:.3f}s")
                
            frame_count += 1
            
            # Wait until the next second
            next_time = start_time + frame_count
            current_time = time.time()
            if current_time < next_time:
                time.sleep(next_time - current_time)
                
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    
    print(f"Finished processing {frame_count} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YouTube stream and get object detections")
    parser.add_argument("youtube_url", help="YouTube URL")
    parser.add_argument("--cookies", help="Path to Netscape/Mozilla cookies.txt file")
    parser.add_argument("--api", default="http://localhost:8000/detect/", help="Detection API URL")
    parser.add_argument("--split", type=int, default=4, help="Split number for detection")
    parser.add_argument("--confidence", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--parallel", type=lambda x: x.lower() == 'true', default=True, 
                      help="Use parallel processing (true/false)")
    parser.add_argument("--source-id", help="Source ID for the stream (default: derived from URL)")
    
    args = parser.parse_args()
    
    # If source_id not provided, derive it from the URL
    if not args.source_id:
        # Extract video ID from YouTube URL
        if "v=" in args.youtube_url:
            args.source_id = args.youtube_url.split("v=")[1].split("&")[0]
        else:
            args.source_id = args.youtube_url.split("/")[-1]
    
    process_stream(
        args.youtube_url,
        args.cookies,
        args.api,
        args.split,
        args.confidence,
        args.parallel,
        args.source_id
    )