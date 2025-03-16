#!/usr/bin/env python3
"""
HLS Stream Worker

This worker:
1. Listens for new frames via Redis pub/sub
2. Maintains a buffer of frames for each source
3. Creates HLS segments and playlists
4. Serves the HLS content via a simple HTTP server with CORS support
"""

import os
import cv2
import numpy as np
import json
import time
import redis
import threading
import subprocess
import logging
import shutil
import http.server
import socketserver
from datetime import datetime
from pathlib import Path
import argparse
import base64
import socket
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hls_stream_worker')

# Configuration (can be overridden by environment variables)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_CHANNEL = os.getenv('REDIS_CHANNEL', 'frame_updates')
HTTP_PORT = int(os.getenv('HTTP_PORT', '8080'))
HLS_DIR = os.getenv('HLS_DIR', 'hls_streams')
SEGMENT_DURATION = int(os.getenv('SEGMENT_DURATION', '30'))  # seconds
REFRESH_INTERVAL = int(os.getenv('REFRESH_INTERVAL', '15'))  # seconds
FRAME_RATE = int(os.getenv('FRAME_RATE', '1'))  # FPS
MAX_SEGMENTS = int(os.getenv('MAX_SEGMENTS', '5'))  # Number of segments to keep
FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'ffmpeg')

# Create the HLS directory
os.makedirs(HLS_DIR, exist_ok=True)

# Custom request handler with CORS support
class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Origin, Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

# Frame buffer for each source ID
frame_buffers = {}
buffer_locks = {}

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket connection to a public server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def start_http_server():
    """Start the HTTP server to serve HLS content"""
    socketserver.TCPServer.allow_reuse_address = True

    # Change directory to HLS root
    os.chdir(HLS_DIR)
    
    # Create and start the server
    handler = CORSHTTPRequestHandler
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", HTTP_PORT), handler)
    
    # local_ip = get_local_ip()
    local_ip = "0.0.0.0"
    logger.info(f"HTTP Server running at http://{local_ip}:{HTTP_PORT}/")
    logger.info(f"Stream URLs will be: http://{local_ip}:{HTTP_PORT}/[source_id]/index.m3u8")
    
    # Run the server in a thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return httpd

def create_source_directory(source_id):
    """Create directory for a source if it doesn't exist"""
    source_dir = os.path.join(HLS_DIR, source_id)
    os.makedirs(source_dir, exist_ok=True)
    return source_dir

def add_frame_to_buffer(source_id, frame, timestamp):
    """Add a frame to the buffer for a specific source"""
    if source_id not in frame_buffers:
        frame_buffers[source_id] = []
        buffer_locks[source_id] = threading.Lock()
        # Create directory for this source
        create_source_directory(source_id)
    
    with buffer_locks[source_id]:
        # Add new frame
        frame_buffers[source_id].append({
            'frame': frame,
            'timestamp': timestamp
        })
        
        # Keep only the last 30 seconds worth of frames
        max_frames = SEGMENT_DURATION * FRAME_RATE
        if len(frame_buffers[source_id]) > max_frames:
            frame_buffers[source_id] = frame_buffers[source_id][-max_frames:]

def create_hls_segment(source_id):
    """Create an HLS segment from the current frame buffer"""
    with buffer_locks.get(source_id, threading.Lock()):
        buffer = frame_buffers.get(source_id, [])
        
        if not buffer:
            logger.warning(f"No frames in buffer for {source_id}, skipping segment creation")
            return None
            
        # Create temp directory for frames
        temp_dir = os.path.join(HLS_DIR, source_id, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save frames as images
        for i, frame_data in enumerate(buffer):
            frame = frame_data['frame']
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(frame_path, frame)
        
        # Create a file list for FFmpeg
        list_path = os.path.join(temp_dir, "frames.txt")
        with open(list_path, 'w') as f:
            for i in range(len(buffer)):
                f.write(f"file 'frame_{i:05d}.jpg'\n")
                f.write(f"duration {1.0/FRAME_RATE}\n")
            
            # Add the last frame one more time to ensure correct duration
            if buffer:
                f.write(f"file 'frame_{len(buffer)-1:05d}.jpg'\n")
        
        # Create segment filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        segment_filename = f"segment_{timestamp}.ts"
        segment_path = os.path.join(HLS_DIR, source_id, segment_filename)
        
        # Use FFmpeg to create segment
        try:
            cmd = [
                FFMPEG_PATH,
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-g', str(FRAME_RATE * 2),
                '-pix_fmt', 'yuv420p',
                '-f', 'mpegts',
                segment_path
            ]
            
            # Run FFmpeg
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            
            logger.info(f"Created HLS segment for {source_id}: {segment_filename}")
            
            # Clean up temp files
            shutil.rmtree(temp_dir)
            
            return segment_filename
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            return None
        except Exception as e:
            logger.error(f"Error creating segment: {str(e)}")
            return None

def update_playlist(source_id, new_segment=None):
    """Update the HLS playlist for a source"""
    source_dir = os.path.join(HLS_DIR, source_id)
    playlist_path = os.path.join(source_dir, "index.m3u8")
    
    # Get all existing segment files
    segments = [f for f in os.listdir(source_dir) if f.endswith('.ts')]
    segments.sort()  # Sort by name (which includes timestamp)
    
    # Add the new segment if provided
    if new_segment and new_segment not in segments:
        segments.append(new_segment)
    
    # Keep only the most recent segments
    segments = segments[-MAX_SEGMENTS:]
    
    # Create playlist content
    playlist_content = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{SEGMENT_DURATION}",
        f"#EXT-X-MEDIA-SEQUENCE:{len(segments)}",
        "#EXT-X-PLAYLIST-TYPE:EVENT"
    ]
    
    for segment in segments:
        playlist_content.append(f"#EXTINF:{SEGMENT_DURATION}.0,")
        playlist_content.append(segment)
    
    # Write playlist file
    with open(playlist_path, 'w') as f:
        f.write('\n'.join(playlist_content))
    
    logger.info(f"Updated playlist for {source_id} with {len(segments)} segments")
    
    # Clean up old segments (keep only those in the playlist plus a few extras)
    all_segments = [f for f in os.listdir(source_dir) if f.endswith('.ts')]
    segments_to_keep = set(segments)
    
    for segment in all_segments:
        if segment not in segments_to_keep:
            try:
                os.remove(os.path.join(source_dir, segment))
                logger.debug(f"Removed old segment: {segment}")
            except Exception as e:
                logger.error(f"Error removing segment {segment}: {str(e)}")

def segment_creation_loop():
    """Loop to periodically create new segments"""
    while True:
        try:
            # Process each source with frames
            for source_id in list(frame_buffers.keys()):
                with buffer_locks.get(source_id, threading.Lock()):
                    # Skip if no frames
                    if not frame_buffers.get(source_id):
                        continue
                
                # Create a new segment
                new_segment = create_hls_segment(source_id)
                
                # Update playlist
                if new_segment:
                    update_playlist(source_id, new_segment)
            
            # Sleep until next refresh
            time.sleep(REFRESH_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in segment creation loop: {str(e)}")
            time.sleep(5)  # Sleep a bit on error before trying again

def process_frame_message(message):
    """Process a frame message from Redis"""
    try:
        # Decode the message
        data = json.loads(message)
        
        # Extract fields
        source_id = data.get('source_id')
        timestamp = data.get('timestamp')
        frame_data = data.get('frame')
        
        if not all([source_id, timestamp, frame_data]):
            logger.warning(f"Incomplete frame message: {message[:100]}...")
            return
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.warning(f"Failed to decode frame for {source_id}")
            return
        
        # Add to buffer
        add_frame_to_buffer(source_id, frame, timestamp)
        logger.info(f"Added frame for {source_id}, timestamp: {timestamp}")
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON message: {message[:100]}...")
    except Exception as e:
        logger.error(f"Error processing frame message: {str(e)}")

def redis_listener():
    """Listen for frame messages on Redis channel"""
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    pubsub = redis_client.pubsub()
    
    pubsub.subscribe(REDIS_CHANNEL)
    logger.info(f"Listening for frame updates on Redis channel: {REDIS_CHANNEL}")
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            process_frame_message(message['data'])

def main():
    """Main function to run the HLS stream worker"""
    logger.info("Starting HLS Stream Worker")
    
    # Start HTTP server
    httpd = start_http_server()
    
    # Start segment creation loop in a thread
    segment_thread = threading.Thread(target=segment_creation_loop)
    segment_thread.daemon = True
    segment_thread.start()
    
    # Start Redis listener in main thread
    try:
        redis_listener()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        httpd.shutdown()
    except Exception as e:
        logger.error(f"Error in Redis listener: {str(e)}")
        httpd.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLS Stream Worker")
    parser.add_argument("--redis-host", help="Redis host", default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, help="Redis port", default=REDIS_PORT)
    parser.add_argument("--redis-channel", help="Redis channel", default=REDIS_CHANNEL)
    parser.add_argument("--http-port", type=int, help="HTTP server port", default=HTTP_PORT)
    parser.add_argument("--hls-dir", help="HLS directory", default=HLS_DIR)
    parser.add_argument("--segment-duration", type=int, help="Segment duration in seconds", default=SEGMENT_DURATION)
    parser.add_argument("--refresh-interval", type=int, help="Segment refresh interval in seconds", default=REFRESH_INTERVAL)
    
    args = parser.parse_args()
    
    # Update config from args
    REDIS_HOST = args.redis_host
    REDIS_PORT = args.redis_port
    REDIS_CHANNEL = args.redis_channel
    HTTP_PORT = args.http_port
    HLS_DIR = args.hls_dir
    SEGMENT_DURATION = args.segment_duration
    REFRESH_INTERVAL = args.refresh_interval
    
    main()
