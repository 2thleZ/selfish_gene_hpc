import imageio.v2 as imageio
import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def make_video(frame_dir, output_file, fps, zoom_factor):
    print(f"Searching for frames in {frame_dir}...")
    files = sorted(glob.glob(os.path.join(frame_dir, "*.ppm")))
    
    if not files:
        print("No frames found!")
        return

    print(f"Found {len(files)} frames. Stitching video to {output_file} with {zoom_factor}x zoom...")
    
    with imageio.get_writer(output_file, fps=fps) as writer:
        for i, filename in enumerate(files):
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(files)}", end='\r')
            
            # Read image
            # Read image
            image_data = imageio.imread(filename)
            
            # Extract step number from filename (e.g. step_000120.ppm -> 120)
            base_name = os.path.basename(filename)
            try:
                 step_num = int(''.join(filter(str.isdigit, base_name)))
            except ValueError:
                 step_num = i
            
            # Always convert to PIL to draw text (and optionally zoom)
            pil_img = Image.fromarray(image_data)
            draw = ImageDraw.Draw(pil_img)
            
            if zoom_factor > 1:
                width, height = pil_img.size
                
                # Calculate Crop Box (Center)
                vis_w = width // zoom_factor
                vis_h = height // zoom_factor
                
                left = (width - vis_w) // 2
                top = (height - vis_h) // 2
                right = left + vis_w
                bottom = top + vis_h
                
                # Crop and Resize (Nearest Neighbor)
                pil_img = pil_img.crop((left, top, right, bottom))
                pil_img = pil_img.resize((width, height), resample=Image.NEAREST)
            
            # Draw Text AFTER zoom so it's always visible in the corner
            draw = ImageDraw.Draw(pil_img)
            text = f"Gen: {step_num}"
            
            # To make text "larger" without needing system fonts, we can:
            # 1. Draw small text
            # 2. Scale up (manually or just find a font)
            # Since we can't reliably find .ttf files, we'll brute-force a pixel art scale up:
            
            # Temporary canvas for text
            txt_scale = 4 # Make it 4x bigger
            txt_img = Image.new('RGBA', (200, 50), (0,0,0,0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0, 0), text, fill=(255, 255, 255), font=ImageFont.load_default())
            
            # Crop to content
            bbox = txt_img.getbbox()
            if bbox:
                txt_img = txt_img.crop(bbox)
                # Scale up
                new_w = txt_img.width * txt_scale
                new_h = txt_img.height * txt_scale
                txt_img = txt_img.resize((new_w, new_h), resample=Image.NEAREST)
                
                # Paste onto main image with padding
                pil_img.paste(txt_img, (20, 20), txt_img)

            image_data = np.array(pil_img)
            
            writer.append_data(image_data)
    
    print(f"\nDone! Video saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="frames", help="Directory containing frames")
    parser.add_argument("--output", default="simulation.mp4", help="Output filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--zoom", type=int, default=1, help="Zoom factor (e.g. 2, 4)")
    args = parser.parse_args()
    
    make_video(args.frames, args.output, args.fps, args.zoom)
