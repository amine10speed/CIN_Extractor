import sys
import os

# Dynamically add the "utils" directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")
sys.path.append(utils_dir)

# Now import cin_extractor without modifying it
from cin_extractor import process_cin_card

from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_data, annotated_image_path, cropped_resized_path = process_cin_card(file_location)
    except Exception as e:
        os.remove(file_location)
        return {"message": "Error processing the CIN card.", "error": str(e)}

    os.remove(file_location)

    return {
        "message": "CIN Extraction Completed",
        "data": extracted_data,
        "annotated_image": annotated_image_path,
        "cropped_resized_image": cropped_resized_path
    }



@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    # Read the uploaded file as a NumPy array
    file_bytes = await file.read()
    np_image = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Process the image
    try:
        extracted_data, annotated_image_path, cropped_resized_path = process_cin_card(image)
    except Exception as e:
        return {"message": "Error processing CIN card", "error": str(e)}

    return {
        "message": "CIN Extraction Completed",
        "data": extracted_data,
        "annotated_image": annotated_image_path,
        "cropped_resized_image": cropped_resized_path,
    }






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
