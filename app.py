import io
import os
import json
import time
import traceback
from typing import List, Optional
from PIL import Image, ImageOps
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="Grantha Library Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3.6-flash"

class RegisterEntry(BaseModel):
    stock_no: str = Field(default="", description="സ്റ്റോക്ക് നമ്പർ")
    book_title: str = Field(default="", description="പുസ്തകത്തിന്റെ പേര്")
    author_publisher: Optional[str] = Field(default="", description="ഗ്രന്ഥകർത്താവിന്റെയും പ്രസാധകന്റെയും പേര്")
    category: Optional[str] = Field(default="", description="ഏത് വിഭാഗത്തിൽ പെടുന്നു")
    how_obtained: Optional[str] = Field(default="", description="എങ്ങനെ കിട്ടി")
    price: Optional[str] = Field(default="", description="വില")
    bill: Optional[str] = Field(default="", description="ബില്ല്")
    date: Optional[str] = Field(default="", description="തീയതി")
    voucher_posted: Optional[str] = Field(default="", description="പോസ്റ്റ് ചെയ്തത്")
    classification_no: Optional[str] = Field(default="", description="ക്ലാസിഫിക്കേഷൻ നമ്പർ")
    almirah_no: Optional[str] = Field(default="", description="അലമാര നമ്പർ")
    remarks: Optional[str] = Field(default="", description="റിമാർക്സ്")

class PageExtraction(BaseModel):
    entries: List[RegisterEntry]

def prepare_mobile_image(image_bytes: bytes) -> bytes:
    """Corrects phone EXIF rotation and converts to a standard JPEG buffer."""
    img = Image.open(io.BytesIO(image_bytes))
    # Correct orientation from phone camera metadata
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Resize slightly if over 2200px to maintain speed & stay within API limits
    if max(img.size) > 2200:
        img.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()

@app.post("/api/scan")
async def scan_page(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        jpeg_bytes = prepare_mobile_image(raw_bytes)

        prompt = """
        You are an expert archivist transcribing a handwritten Kerala Library Council Stock Register (സ്റ്റോക്ക് രജിസ്റ്റർ).

        COLUMN 2 FOCUS (പുസ്തകത്തിന്റെ പേര്):
        - Transcribe handwritten Malayalam book titles with complete fidelity.
        - Accurately identify chillu letters (ർ, ൽ, ൾ, ൻ, ൺ, ൿ) and conjuncts (ക്ക, ച്ച, ത്ത, പ്പ, ണ്ട, ന്ത, ങ്ക, ഷ്ട, ണ്ണ, ഷ്ണ, etc.).
        - Accurately capture vowel diacritics (ി, ീ, ു, ൂ, ൃ, െ, േ, ൈ, ൊ, ോ, ൌ).
        - If titles are in English, transcribe cleanly.
        - Resolve ditto marks ('"' or '-') by copying down the value from the row directly above.
        """

        for attempt in range(1, 4):
            try:
                print(f"Calling Gemini API for uploaded mobile page (Attempt {attempt})...")
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[
                        types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageExtraction,
                        temperature=0.0
                    )
                )
                print("Extraction successful! Returning rows to phone...")
                return json.loads(response.text)
            except Exception as api_err:
                print(f"API attempt {attempt} failed: {api_err}")
                if attempt < 3:
                    time.sleep(3)
                else:
                    raise api_err

    except Exception as e:
        print("Detailed Server Error Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
async def export_to_excel(entries: List[RegisterEntry]):
    data = [e.dict() for e in entries]
    df = pd.DataFrame(data)
    df.columns = [
        "സ്റ്റോക്ക് നമ്പർ",
        "പുസ്തകത്തിന്റെ പേര്",
        "ഗ്രന്ഥകർത്താവിന്റെയും പ്രസാധകന്റെയും പേര്",
        "ഏത് വിഭാഗത്തിൽ പെടുന്നു",
        "എങ്ങനെ കിട്ടി",
        "വില",
        "ബില്ല്",
        "തീയതി",
        "പോസ്റ്റ് ചെയ്തത്",
        "ക്ലാസിഫിക്കേഷൻ നമ്പർ",
        "അലമാര നമ്പർ",
        "റിമാർക്സ്"
    ]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Stock_Register", index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=library_stock_register.xlsx"}
    )

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        return f.read()