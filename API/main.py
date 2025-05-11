from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
import io
import json
from dotenv import load_dotenv
from loguru import logger
from os import getenv

load_dotenv()

GEMINI_API_KEY = getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

@app.post("/check-freshness/")
async def check_food_freshness(file: UploadFile = File(...)):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))

    prompt = (
        "You are a food quality inspector. Carefully analyze the **visible condition** of each vegetable or perishable item in the image provided. "
        "For each visible vegetable or perishable food item, estimate the number of days left before it spoils based on the following factors:\n"
        "- **Color and Texture**: Look for any discoloration, wrinkles, bruising, or softness.\n"
        "- **Bruising and Mold**: These factors can drastically reduce shelf life.\n"
        "- **Size and Firmness**: Shriveling or softness usually means it's closer to spoilage.\n"
        "- **Standard Shelf Life Adjustments**: For specific items (e.g., leafy greens, potatoes, carrots), adjust based on their usual spoilage patterns.\n\n"
        "The result must be returned in the following JSON format (use actual detected vegetables from the image, not examples):\n\n"
        "{\n"
        "  \"<Vegetable Name>\": {\"days_left\": <Number>, \"observation\": <String>},\n"
        "  \"<Another Vegetable>\": {\"days_left\": <Number>, \"observation\": <String>}\n"
        "}\n"
    )

    response = model.generate_content([prompt, img])

    try:
        response_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        ingredient_dict = json.loads(response_text)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Could not parse model response as JSON."}, status_code=400)

    sorted_dict = dict(sorted(ingredient_dict.items(), key=lambda item: item[1]['days_left']))

    return JSONResponse(content=sorted_dict)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
