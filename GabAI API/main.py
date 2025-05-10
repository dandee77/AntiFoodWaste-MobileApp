import google.generativeai as genai
from PIL import Image
import os


# Configure your Gemini API key
genai.configure(api_key="AIzaSyADjfHTmw-_ovAZQN487UvAnwp3HvXnFXI")

# Initialize the vision model
model = genai.GenerativeModel("gemini-2.0-flash")

def analyze_pantry_image(image_path):
    """
    Takes a picture of a pantry and generates a recipe.
    """
    img = Image.open(image_path)
    prompt = "Generate a recipe using the items you see in this pantry image. Be creative but practical. Ration it into multiple meals and stretch it as long as possible. Use the ingredients that tend to spoil first"

    response = model.generate_content([prompt, img])
    return response.text

def check_food_freshness(image_path):
    """
    Takes a picture of perishable goods and returns a list of visible ingredients,
    sorted by shelf life (shortest to longest) in the format:
    'Ingredient Name: X'
    Where X is the number of days left before it spoils.
    """
    img = Image.open(image_path)
    prompt = (
        "From this image, identify all visible perishable food items. "
        "Return the list sorted in ascending order based on how soon each item will expire. "
        "Use this format exactly: 'Ingredient Name: X' where X is the number of days left (only the number, no units or extra text). "
        "Do not include explanations—just the list."
    )

    response = model.generate_content([prompt, img])
    return response.text




# === EXAMPLE USAGE ===

# Pantry example
pantry_result = check_food_freshness("pantry.jpg")
pantry_result = pantry_result.splitlines()
ingredient_dict = {}
for item in pantry_result:
    name, days_left = item.split(':')
    ingredient_dict[name.strip()] = int(days_left.strip().split()[0])
print("Food Freshness:\n", ingredient_dict)
print("Recipe Suggestion:\n", pantry_result)

# Food freshness example
