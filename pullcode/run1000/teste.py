import pytesseract
from PIL import Image

if __name__ == "__main__":

    img = Image.open("sstes.png")
    tesr = pytesseract.image_to_string(img, lang='chi_sim')
    print(tesr)
