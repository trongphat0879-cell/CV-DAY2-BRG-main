import urllib.request as request
import cv2 as cv
import numpy as np

def read_image_from_github(url):
    resp = request.urlopen(url)
    arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    return img

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    img = read_image_from_github(url)

    if img is None:
        print("Không đọc được ảnh")
    else:
        cv.imshow("Lena", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
