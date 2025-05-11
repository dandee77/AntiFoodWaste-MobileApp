from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
import io
import json

# Configure Gemini API
genai.configure(api_key="AIzaSyADjfHTmw-_ovAZQN487UvAnwp3HvXnFXI")
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()


@app.post("/analyze-pantry/")
async def analyze_pantry_image(file: UploadFile = File(...)):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    
    prompt = (
        "From this pantry image, identify all usable ingredients and generate 2–3 simple recipes. "
        "Focus on ingredients likely to expire soon. "
        "Return result in **valid JSON** exactly like this:\n\n"
        "{\n"
        "  \"Recipe Name 1\": {\n"
        "    \"Step 1\": \"...\",\n"
        "    \"Step 2\": \"...\"\n"
        "  },\n"
        "  \"Recipe Name 2\": {\n"
        "    \"Step 1\": \"...\",\n"
        "    \"Step 2\": \"...\"\n"
        "  }\n"
        "}\n"
        "DO NOT include any explanation or markdown—just return raw JSON."
    )

    response = model.generate_content([prompt, img])

    try:
        response_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        recipe_dict = json.loads(response_text)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Could not parse model response as JSON."}, status_code=400)

    return JSONResponse(content={"recipes": recipe_dict})


@app.post("/check-freshness/")
async def check_food_freshness(file: UploadFile = File(...)):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))

    prompt = (
        "You are a food quality inspector. Carefully analyze the **visual condition** of each vegetable or perishable item in this image. "
        "Estimate the number of days left before each one spoils based on visible signs of aging or rot such as discoloration, bruising, mold, wrinkles, or softness. "
        "Additionally, provide a confidence score for each prediction, indicating how certain you are about the freshness estimate. "
        "The confidence score should be a number between 0 and 100, where 100 means highly confident in the prediction. "
        "Return the result in this **strict JSON format** with no markdown, text, or commentary:\n\n"
        "{\n"
        "  \"Potato\": {\"days_left\": 2, \"confidence\": 85},\n"
        "  \"Spinach\": {\"days_left\": 1, \"confidence\": 90},\n"
        "  \"Carrot\": {\"days_left\": 5, \"confidence\": 80}\n"
        "}"
    )

    response = model.generate_content([prompt, img])

    try:
        response_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        ingredient_dict = json.loads(response_text)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Could not parse model response as JSON."}, status_code=400)

    # Sort the results by days left, ascending, and return the JSON
    sorted_dict = dict(sorted(ingredient_dict.items(), key=lambda item: item[1]['days_left']))

    return JSONResponse(content=sorted_dict)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
