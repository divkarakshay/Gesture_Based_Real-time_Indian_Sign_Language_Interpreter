import numpy as np
import cv2
# boundaries = [
#   ([0, 120, 0], [140, 255, 100]),
#   ([25, 0, 75], [180, 38, 255])
#]
boundaries = [
    ([42, 47, 89], [180, 188, 236]),
    ([36, 85, 141], [125, 194, 241])
]


def handsegment(frame):
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)

    # for i,(lower, upper) in enumerate(boundaries):
    #   # create NumPy arrays from the boundaries
    #   lower = np.array(lower, dtype = "uint8")
    #   upper = np.array(upper, dtype = "uint8")

    #   # find the colors within the specified boundaries and apply
    #   # the mask
    #   if(i==0):
    #       print "Harish"
    #       mask1 = cv2.inRange(frame, lower, upper)
    #   else:
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    # 		print "Aadi"
    # 		mask2 = cv2.inRange(frame, lower, upper)
    mask = cv2.bitwise_or(mask1, mask2)
    out = cv2.bitwise_and(frame, frame, mask=mask)
    output = cv2.resize(out, (1920,1080))
    # show the images
    # cv2.imshow("images", mask)
    # cv2.imshow("images", output)
    return output

if __name__ == '__main__':
    frame = cv2.imread("test.jpeg")
    handsegment(frame)
