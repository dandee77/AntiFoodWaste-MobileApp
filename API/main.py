from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
import io

# Configure Gemini API
genai.configure(api_key="AIzaSyADjfHTmw-_ovAZQN487UvAnwp3HvXnFXI")
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

import json

@app.post("/analyze-pantry/")
async def analyze_pantry_image(file: UploadFile = File(...)):
    """
    Receives an image and generates detailed recipes using pantry items.
    Returns a dictionary where the key is the dish name and the value is a dictionary of detailed steps.
    """
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    
    prompt = (
        "From this pantry image, identify all usable ingredients and generate multiple recipes. "
        "Each recipe should use ingredients that are likely to spoil first. "
        "For each recipe, break it down into clear and detailed steps, including exact measurements, cooking methods, and timing. "
        "Return the result in valid JSON format like this:\n\n"
        "{\n"
        "  \"Recipe Name 1\": {\n"
        "    \"Step 1\": \"instruction...\",\n"
        "    \"Step 2\": \"instruction...\"\n"
        "  },\n"
        "  \"Recipe Name 2\": {\n"
        "    \"Step 1\": \"instruction...\",\n"
        "    \"Step 2\": \"instruction...\"\n"
        "  }\n"
        "}\n"
        "Do not include any explanation or extra text—just return a valid JSON object."
    )

    response = model.generate_content([prompt, img])
    
    # Attempt to parse the output as JSON
    try:
        # Clean common formatting mistakes if needed
        response_text = response.text.strip().strip("```json").strip("```").strip()
        recipe_dict = json.loads(response_text)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Could not parse model response as JSON."}, status_code=400)

    return JSONResponse(content={"recipes": recipe_dict})



@app.post("/check-freshness/")
async def check_food_freshness(file: UploadFile = File(...)):
    """
    Receives an image and returns a sorted dict of perishable ingredients
    and how many days they have left.
    """
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    prompt = (
        "From this image, identify all visible perishable food items. "
        "Return the list sorted in ascending order based on how soon each item will expire. "
        "Use this format exactly: 'Ingredient Name: X' where X is the number of days left (only the number, no units or extra text). "
        "Do not include explanations—just the list."
    )

    response = model.generate_content([prompt, img])
    lines = response.text.strip().splitlines()
    ingredient_dict = {}

    for line in lines:
        if ':' in line:
            name, days = line.split(':', 1)
            try:
                ingredient_dict[name.strip()] = int(days.strip())
            except ValueError:
                continue  # skip if not parsable

    return JSONResponse(content={"freshness": ingredient_dict})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)