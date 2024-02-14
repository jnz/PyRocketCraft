#!/bin/bash

WINDOW_TITLE="Bullet Physics"

source env.sh
python3 src/rocketcraft.py &
PROGRAM_PID=$!

# Wait for the window to appear
WINDOW_ID=""
while [ -z "$WINDOW_ID" ]; do
  sleep 2.0  # Wait a bit before checking again to avoid excessive CPU usage
  WINDOW_ID=$(wmctrl -l | grep "$WINDOW_TITLE" | awk '{print $1}')
done

# Get window geometry
GEOMETRY=$(wmctrl -lG | grep "$WINDOW_TITLE" | awk '{print $5"x"$6"+"$3"+"$4}')

CURRENT_DISPLAY=$DISPLAY

# Start capturing using ffmpeg, running in the background
ffmpeg -f x11grab -framerate 25 -video_size "$(echo $GEOMETRY | awk -F'+' '{print $1}')" -i $CURRENT_DISPLAY+$(echo $GEOMETRY | awk -F'+' '{print $2","$3}') -y output.mp4 &
FFMPEG_PID=$!

# Monitor the program's process. If the program exits, stop the ffmpeg recording.
while kill -0 $PROGRAM_PID 2> /dev/null; do
  sleep 0.1
done

# Stop ffmpeg by sending a 'q' command to gracefully end recording
kill -SIGINT $FFMPEG_PID

# Wait a bit to ensure ffmpeg has time to finish writing the video file
wait $FFMPEG_PID
