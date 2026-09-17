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
    stock_no: str = Field(default="", description="സ്റ്റോക്ക് നമ്പർ (Column 1 numeric accession number)")
    book_title: str = Field(default="", description="പുസ്തകത്തിന്റെ പേര് (Column 2 Book Title)")
    author_publisher: Optional[str] = Field(default="", description="ഗ്രന്ഥകർത്താവിന്റെയും പ്രസാധകന്റെയും പേര് (Column 3 Author/Publisher)")
    category: Optional[str] = Field(default="", description="ഏത് വിഭാഗത്തിൽ പെടുന്നു (Column 4)")
    how_obtained: Optional[str] = Field(default="", description="എങ്ങനെ കിട്ടി (Column 5 e.g. ഗ്രാൻഡ്, വിലയ്ക്ക്)")
    price: Optional[str] = Field(default="", description="വില (Column 6 Price in ₹)")
    bill: Optional[str] = Field(default="", description="ബില്ല് (Column 7 Bill/Voucher)")
    date: Optional[str] = Field(default="", description="തീയതി (Column 8 Date)")
    voucher_posted: Optional[str] = Field(default="", description="പോസ്റ്റ് ചെയ്തത് (Column 9)")
    classification_no: Optional[str] = Field(default="", description="ക്ലാസിഫിക്കേഷൻ നമ്പർ (Column 10)")
    almirah_no: Optional[str] = Field(default="", description="അലമാര നമ്പർ (Column 11)")
    remarks: Optional[str] = Field(default="", description="റിമാർക്സ് (Column 12)")

class PageExtraction(BaseModel):
    entries: List[RegisterEntry]

def fix_image_orientation(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    
    # Auto-rotate to landscape if phone uploaded upright portrait
    if img.height > img.width:
        img = img.rotate(270, expand=True)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    if max(img.size) > 2400:
        img.thumbnail((2400, 2400), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

@app.post("/api/scan")
async def scan_page(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        jpeg_bytes = fix_image_orientation(raw_bytes)

        prompt = """
        You are an expert archivist transcribing a two-page spread of a Kerala Library Council Stock Register (സ്റ്റോക്ക് രജിസ്റ്റർ).

        CRITICAL ROW-MATCHING & ALIGNMENT RULES:
        1. STRICT LINE-BY-LINE ALIGNMENT:
           - Each output row must correspond to ONE continuous printed horizontal line across both left and right pages.
           - Column 1 (സ്റ്റോക്ക് നമ്പർ), Column 2 (പുസ്തകത്തിന്റെ പേര്), and Column 3 (ഗ്രന്ഥകർത്താവ്) MUST come from the exact same row.
           - DO NOT shift authors or titles across rows.
           - If Row 1 has an empty title or was an unused header line, DO NOT pull the author from Row 2 into Row 1.
           - ONLY extract rows that contain an actual book entry. Skip empty rows.

        2. ACCURATE COLUMN VALUES:
           - Column 1: സ്റ്റോക്ക് നമ്പർ (e.g., 26169, 26170, 26171...)
           - Column 2: പുസ്തകത്തിന്റെ പേര് (e.g., 26169 is 'കേസ് ഫയൽസ്', 26170 is 'രാജമുദ്ര കേസ് ഡയറി', 26171 is 'കഥ', 26172 is 'ഒടുക്കം')
           - Column 3: ഗ്രന്ഥകർത്താവിന്റെയും പ്രസാധകന്റെയും പേര് (e.g., for 26169 it is 'ശ്യാം കൃഷ്ണൻ, സി.യു', for 26170 it is 'സുരേന്ദ്രൻ മണ്ണാട്', for 26171 it is 'സാറാ ജോസഫ്')
           - Column 4: വിഭാഗം (e.g., Novel, Essays, ഓർമ്മ, കഥ)
           - Column 5: എങ്ങനെ കിട്ടി (e.g., 'ഗ്രാന്റ്' or 'വിലയ്ക്ക്')
           - Columns 6+: വില (Price), ബില്ല് (Bill), തീയതി (Date), ക്ലാസിഫിക്കേഷൻ നമ്പർ.

        3. DITTO MARKS ('"' or '-'):
           - Expand ditto marks to the text of the valid cell directly above.
        """

        for attempt in range(1, 4):
            try:
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
                return json.loads(response.text)
            except Exception as api_err:
                if attempt < 3:
                    time.sleep(3)
                else:
                    raise api_err

    except Exception as e:
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
