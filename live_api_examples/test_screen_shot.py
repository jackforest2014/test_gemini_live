import os
from google import genai
import time

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

my_file = client.files.upload(file="../data/1.png")

start_time = time.time()
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[my_file, "Extract problem statements and the whole code from this image, also extract the selected line of code (has different background color than other codes)."],
)
end_time = time.time()
print(f"Content generation took {end_time - start_time:.2f} seconds")

# print(response.text)


start_time = time.time()
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[my_file, "Extract the selected line of code (has different background color than other codes)."],
)
end_time = time.time()
print(f"Content generation took {end_time - start_time:.2f} seconds")

print(response.text)