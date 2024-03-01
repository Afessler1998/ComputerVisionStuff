import cv2

if __name__ == '__main__':
    img = cv2.imread('img.png')
    print(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('img2.png', img)