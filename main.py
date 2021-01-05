# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import Image
import cv2
import pytesseract

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    img = cv2.imread('./image.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_inv, (x - 2, y - 2), (x + w + 1, y + h + 1), (255, 77, 77), 1)
    cv2.imwrite('./image_gray.jpg', img_inv)
    # text = pytesseract.image_to_string(img_gray, lang="eng")
    # print(text)
    img = Image.open('./image_gray.jpg')
    img.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
