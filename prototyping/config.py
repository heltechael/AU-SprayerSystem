import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
PORT_NAME = "COM3"
BAUD_RATE = 9600
TIMEOUT = 1

NUM_NOZZLES = 25
IMAGE_SIZE = (4160, 2360)
WEED_LIST = ["oneWeed", "twoWeed", "threeWeed", "fourWeed", "fiveWeed", "sixWeed", "sevenWeed", "eightWeed",
             "nineWeed", "tenWeed", "elevenWeed", "twelveWeed", "thirteenWeed", "fourteenWeed", "fifteenWeed"]
CROP_LIST = ["oneCrop", "twoCrop", "threeCrop", "fourCrop", "fiveCrop", "sixCrop", "sevenCrop", "eightCrop",
             "nineCrop", "tenCrop", "elevenCrop", "twelveCrop", "thirteenCrop", "fourteenCrop", "fifteenCrop"]


def yolo_v_11():

    boxes = generate_random_bounding_boxes(num_boxes=500)
    return boxes


def generate_random_bounding_boxes(image_width=config.IMAGE_SIZE[0], image_height=config.IMAGE_SIZE[1], num_boxes=1):

    return [
        [random.choice(["oneWeed", "oneCrop"]),  # Class ID
         random.randint(100, image_width - 100),  # xc
         random.randint(100, image_height - 100),  # yc
         random.randint(20, 150),  # width
         random.randint(20, 100)]  # height
        for _ in range(num_boxes)
    ]
