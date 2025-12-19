#!/bin/bash
# Convert frames to video
# Usage: ./make_video.sh [output_filename]

OUTPUT=${1:-"simulation.mp4"}

if [ ! -d "frames" ]; then
    echo "Error: frames directory not found."
    exit 1
fi

echo "Converting frames to $OUTPUT..."
ffmpeg -y -framerate 30 -i frames/step_%06d.ppm -c:v libx264 -pix_fmt yuv420p $OUTPUT

echo "Done."
