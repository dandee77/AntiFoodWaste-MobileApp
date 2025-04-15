from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from PIL import Image
import io

#test 

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tourist Attraction Analyzer",
    description="API that analyzes images of tourist attractions using Google Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

@app.post("/analyze-attraction/", response_class=JSONResponse)
async def analyze_tourist_attraction(image: UploadFile = File(...)):
    """
    Analyze an image of a tourist attraction and return its history and details.
    
    Args:
        image: UploadFile containing the image to analyze
        
    Returns:
        JSON response with attraction details or error message
    """
    try:
        # Check if the file is an image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"Processing image: {image.filename}")
        
        # Read the image file and convert to PIL Image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # Prepare the prompt
        prompt = """
        Analyze this image and identify if it shows a known tourist attraction. 
        If it is a tourist attraction, provide the following details in JSON format:
        {
            "name": "Official name of the attraction",
            "location": "City and country where it's located",
            "type": "Type of attraction (e.g., historical site, museum, natural wonder)",
            "year_built": "When it was built or discovered",
            "historical_significance": "Brief history and why it's significant",
            "architectural_style": "If applicable, the architectural style",
            "interesting_facts": ["Array", "of", "interesting", "facts"],
            "recognition": "Any UNESCO or other recognition it has received",
            "visitor_information": "Typical visitor numbers or best times to visit"
        }
        
        If the image doesn't contain a recognizable tourist attraction, return:
        {
            "error": "No recognized tourist attraction in the image"
        }
        """
        
        # Call Gemini API with the PIL Image
        response = model.generate_content([prompt, img])
        
        # Process the response
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
        
        # Try to parse the response (Gemini should return JSON as a string)
        try:
            # The response might include markdown formatting, so we need to clean it
            json_str = response.text.strip().replace('```json', '').replace('```', '').strip()
            return JSONResponse(content=eval(json_str))
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return JSONResponse(content={"response": response.text})
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Tourist Attraction Analyzer API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)