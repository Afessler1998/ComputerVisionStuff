import cv2

if __name__ == '__main__':
    img = cv2.imread('../img.png')
    print(img.shape)

    cropped = img[256:768, 256:768] # y1:y2, x1:x2
    cv2.imshow('image', cropped)
    cv2.waitKey(0)

    resized = cv2.resize(img, (512, 512))
    cv2.imshow('image', resized)
    cv2.waitKey(0)

    flippedVert = cv2.flip(img, 0) # flip vertically
    cv2.imshow('image', flippedVert)
    cv2.waitKey(0)
    flippedHor = cv2.flip(img, 1) # flip horizontally
    cv2.imshow('image', flippedHor)
    cv2.waitKey(0)
    flippedBoth = cv2.flip(img, -1) # flip both vertically and horizontally
    cv2.imshow('image', flippedBoth)
    cv2.waitKey(0)

    center = (img.shape[1]//2, img.shape[0]//2)
    rotMatrix = cv2.getRotationMatrix2D(center, 90, 1.0) # origin, angle, scale
    rotated = cv2.warpAffine(img, rotMatrix, (img.shape[1], img.shape[0]))
    cv2.imshow('image', rotated)
    cv2.waitKey(0)

    cv2.destroyAllWindows()