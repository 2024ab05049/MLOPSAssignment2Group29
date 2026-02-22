"""
Generate a small sample dataset for testing the pipeline.
Creates synthetic cat and dog images for quick testing.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random

def create_sample_image(class_name, index, output_dir):
    """Create a sample image with text and color."""
    # Random colors for variety
    if class_name == 'cat':
        colors = [(255, 200, 150), (200, 180, 160), (220, 190, 170)]
    else:
        colors = [(150, 200, 255), (160, 180, 200), (170, 190, 220)]

    color = random.choice(colors)

    # Create image
    img = Image.new('RGB', (224, 224), color=color)
    draw = ImageDraw.Draw(img)

    # Add some random shapes
    for _ in range(5):
        x1 = random.randint(0, 200)
        y1 = random.randint(0, 200)
        x2 = x1 + random.randint(10, 50)
        y2 = y1 + random.randint(10, 50)
        shape_color = tuple([c + random.randint(-30, 30) for c in color])
        draw.ellipse([x1, y1, x2, y2], fill=shape_color)

    # Add text
    text = f"{class_name.upper()}\n{index}"
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (224 - text_width) // 2
    y = (224 - text_height) // 2

    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    # Save image
    filename = f"{class_name}_{index:04d}.jpg"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)

    return filepath


def generate_sample_dataset(output_dir='data/raw', num_samples_per_class=50):
    """Generate a sample dataset."""
    print("=" * 50)
    print("Generating Sample Dataset")
    print("=" * 50)

    # Create directories
    cat_dir = os.path.join(output_dir, 'cat')
    dog_dir = os.path.join(output_dir, 'dog')

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    print(f"\nGenerating {num_samples_per_class} samples per class...")

    # Generate cat images
    print(f"\nGenerating cat images...")
    for i in range(num_samples_per_class):
        create_sample_image('cat', i, cat_dir)
        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{num_samples_per_class} cat images")

    # Generate dog images
    print(f"\nGenerating dog images...")
    for i in range(num_samples_per_class):
        create_sample_image('dog', i, dog_dir)
        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{num_samples_per_class} dog images")

    print("\n" + "=" * 50)
    print("Sample Dataset Generated!")
    print("=" * 50)
    print(f"\nLocations:")
    print(f"  Cats: {cat_dir}")
    print(f"  Dogs: {dog_dir}")
    print(f"\nTotal images: {num_samples_per_class * 2}")
    print("\nNote: This is a synthetic dataset for testing only.")
    print("For real training, download the Kaggle Cats vs Dogs dataset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory for images')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples per class')

    args = parser.parse_args()

    generate_sample_dataset(args.output_dir, args.num_samples)
