import google.generativeai as genai
import cv2
import os
import numpy as np
from PIL import Image 

def extract_text_image(image_path):


    # Load and process the image

    file_bytes = np.asarray(bytearray(image_path.read()), dtype = np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # to convert BGR to RGB
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # to convert BGR to Grey
    _, image_bw = cv2.threshold(image_grey,150, 255, cv2.THRESH_BINARY) # TO CONVERT GREY TO BLACK N WHITE


    # The image that CV2 gives is in numpy array format, we need to convert it to image object
    final_image = Image.fromarray(image_bw)

    # configure the genai model


    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Lets write prompt for OCR
        
    prompt = '''You act as an OCR application on the fiven image and extract the text from it.
    Give only the text  as output, do not give any other explanation or description'''


    #Lets extract and the image :
    response = model.generate_content([prompt,final_image])
    output_text = response.text
    return output_text