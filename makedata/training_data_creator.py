from PIL import Image, ImageDraw, ImageFont
import random

# Load resources
font_path = "EuroPlate.ttf"
font_size = 145
font = ImageFont.truetype(font_path, font_size)
image = Image.open("base_img.png").convert("RGBA")
draw = ImageDraw.Draw(image)
image_width = image.width

# Spacing
char_spacing = 5
group_spacing = 20
end_padding = 15
y = 15  # vertical position

# Generate plate parts
city_code = random.choice(['B', 'M', 'F', 'S', 'K'])
letter_group = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(1, 2)))
number_group = str(random.randint(1, 9999)).zfill(random.randint(1, 4))

# ---------- Step 1: Calculate width of right-aligned group ----------
def get_text_width(text, font, spacing):
    width = 0
    for i, char in enumerate(text):
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        width += char_width
        if i < len(text) - 1:
            width += spacing
    return width

# Width of second group (letter_group + number_group)
second_group = letter_group + number_group
second_group_width = get_text_width(second_group, font, char_spacing)

# Start X so it ends exactly at (image.width - 15)
start_x_second_group = image_width - end_padding - second_group_width

# ---------- Step 2: Draw city code ----------
x = 80  # or whatever left margin you want
for i, char in enumerate(city_code):
    bbox = draw.textbbox((0, 0), char, font=font)
    char_width = bbox[2] - bbox[0]
    draw.text((x, y), char, font=font, fill="black")
    x += char_width + char_spacing

# Add space between city_code and second group
x += group_spacing

# Optional: add enforcement that `x < start_x_second_group` or adjust dynamically

# ---------- Step 3: Draw second group at aligned position ----------
x = start_x_second_group
for i, char in enumerate(second_group):
    bbox = draw.textbbox((0, 0), char, font=font)
    char_width = bbox[2] - bbox[0]
    draw.text((x, y), char, font=font, fill="black")
    x += char_width + char_spacing

# Save
image.save("plate_right_aligned.png")
