import os
import math
from PIL import Image, ImageDraw, ImageFont

# Set paths
base_dir = r"c:\Users\csunh\Documents\ScourtCaptcha"
training_dir = os.path.join(base_dir, "TrainingData")
grids_dir = os.path.join(base_dir, "grids")
os.makedirs(grids_dir, exist_ok=True)

# 1. Get and sort all PNG files numerically
all_files = [f for f in os.listdir(training_dir) if f.lower().endswith('.png')]
all_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

print(f"Found {len(all_files)} PNG files in TrainingData.")
print(f"Sorted range: {all_files[0]} to {all_files[-1]}")

# Save the sorted filenames to a text file for reference
sorted_list_path = os.path.join(base_dir, "sorted_filenames.txt")
with open(sorted_list_path, "w") as f:
    for filename in all_files:
        f.write(f"{filename}\n")
print(f"Sorted filenames list saved to {sorted_list_path}")

# 2. Grid Parameters
COLS = 5
ROWS = 20
CELL_WIDTH = 180
CELL_HEIGHT = 60
GRID_SIZE = COLS * ROWS # 100

PAGE_WIDTH = COLS * CELL_WIDTH
PAGE_HEIGHT = ROWS * CELL_HEIGHT

# Load Font
try:
    font = ImageFont.truetype("arial.ttf", 14)
except IOError:
    font = ImageFont.load_default()

num_pages = math.ceil(len(all_files) / GRID_SIZE)

for page_idx in range(num_pages):
    start_idx = page_idx * GRID_SIZE
    end_idx = min(start_idx + GRID_SIZE, len(all_files))
    page_files = all_files[start_idx:end_idx]
    
    # Create canvas
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color="white")
    draw = ImageDraw.Draw(canvas)
    
    for idx, filename in enumerate(page_files):
        c = idx % COLS
        r = idx // COLS
        
        x_start = c * CELL_WIDTH
        y_start = r * CELL_HEIGHT
        
        # Draw light gray border for cell structure
        draw.rectangle(
            [x_start, y_start, x_start + CELL_WIDTH - 1, y_start + CELL_HEIGHT - 1],
            outline="#D3D3D3"
        )
        
        # Draw the label (e.g. "0000.png")
        label = filename
        # Draw text vertically centered inside cell
        draw.text((x_start + 5, y_start + 22), label, fill="black", font=font)
        
        # Load and paste the captcha image
        img_path = os.path.join(training_dir, filename)
        with Image.open(img_path) as img:
            # Resize if necessary, but captchas are 121x46. Center it inside the remaining space.
            # Space for image: x from x_start + 55 to x_start + 176 (121px wide)
            # y from y_start + 7 to y_start + 53 (46px high)
            canvas.paste(img, (x_start + 55, y_start + 7))
            
    # Save the page
    grid_filename = f"page_{page_idx:02d}.png"
    grid_path = os.path.join(grids_dir, grid_filename)
    canvas.save(grid_path)
    print(f"Generated grid: {grid_path} (Contains {len(page_files)} images)")

print("All grid pages generated successfully!")
