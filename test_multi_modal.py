import pathlib
import textwrap

import google.generativeai as genai

image_path = "/Users/wangbenlin/Desktop/1.png"  # Replace with the path to your image file

# Configure the API key
genai.configure(api_key='AIzaSyCropygAB_coiTtNaqygE1DLRE1Ubf_nws')

# Load the Gemini Pro Vision model
model = genai.GenerativeModel('models/gemini-2.0-flash-live-001')


def generate_content_with_image_and_text(image_path: str, prompt: str):
    img = pathlib.Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")

    image_parts = [{
        "mime_type": "image/jpeg",  # or image/png
        "data": pathlib.Path(image_path).read_bytes()
    }]

    contents = [prompt, image_parts]
    # print(contents[0])
    response = model.generate_content(contents, stream=True)
    # print(type(response))
    print(response.text)
    # print(response.text)


# Example Usage
# image_path = 'path/to/your/image.jpg'  # Replace with your image path
prompt = 'Summarize what is shown in this image.'
prompt = "don't say anything, just output the summary"

generate_content_with_image_and_text(image_path, prompt)

