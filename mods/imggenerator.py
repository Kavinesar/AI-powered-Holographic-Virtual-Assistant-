import requests
import io
from PIL import Image
import matplotlib.pyplot as plt

API_TOKEN = ''
API_URL = ""  # flux

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")

def display_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    while True:
        prompt = input("Enter an image description (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting the program.")
            break

        print("Generating image ...")
        try:
            image_bytes = query({"inputs": prompt})
            display_image(image_bytes)
            print("Image generation is completed.")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
