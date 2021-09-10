import os
import cv2
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches



def line_select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    obstacleCorners ={"firstCorner":{"x":int(x1),"y":int(y1)},
                      "secondCorner":{"x":int(x2),"y":int(y2)}}
    jsonResult = os.path.join(currentDir, "result.json")
    with open(jsonResult, 'w') as fp:
        json.dump(obstacleCorners, fp)
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
def toggle_selector(event):
    print(' Key pressed.')
    if event.key == 't':
        if toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        else:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

def findArea(curntDir):
    global currentDir
    currentDir = curntDir
    leftImagePath = os.path.join(currentDir, "1.png")
    rightImagePath = os.path.join(currentDir, "2.png")
    if not (os.path.exists(leftImagePath) or os.path.exists(rightImagePath)):
        print("left or right or both images were not exist.")
        return
    leftImage = cv2.imread(leftImagePath)
    rightImage = cv2.imread(rightImagePath)
    fig, ax = plt.subplots()
    ax.imshow(rightImage,'gray')
    ax.set_title(
        "Click and drag to draw a rectangle.\n"
        "Press 't' to toggle the selector on and off.")

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # disable middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()
    cropedDir = os.path.join(currentDir,'croped')
    cropedFiles = os.listdir(cropedDir)
    for filename in cropedFiles:
        cropedImage = cv2.imread(os.path.join(cropedDir,filename),cv2.COLOR_BGR2GRAY)
        figCroped, axCroped = plt.subplots()
        axCroped.imshow(cropedImage)
        f = open(os.path.join(currentDir, "result.json"))
        data = json.load(f)
        print(data["secondCorner"]["x"])
        minXCorner = min(data["secondCorner"]["x"], data["firstCorner"]["x"])
        minYCorner = min(data["secondCorner"]["y"], data["firstCorner"]["y"])
        deltaX = abs(data["secondCorner"]["x"]-data["firstCorner"]["x"])
        deltaY = abs(data["secondCorner"]["y"]-data["firstCorner"]["y"])
        rect = patches.Rectangle((min(data["secondCorner"]["x"], data["firstCorner"]["x"]), min(data["secondCorner"]["y"], data["firstCorner"]["y"])), deltaX, deltaY, linewidth=1, edgecolor='r', facecolor="none")
        axCroped.add_patch(rect)
        temp_result = findNumberOfCorrectDetection(minXCorner, minYCorner, deltaX, deltaY, cropedImage )
        temp_result["fileName"] = os.path.splitext(filename)[0]
        print(temp_result)
        json_result = os.path.join(currentDir,os.path.splitext(filename)[0]+".json")
        with open(json_result, 'w') as fp:
            json.dump(temp_result, fp)
        plt.show()


def findNumberOfCorrectDetection(minXCorner, minYCorner, deltaX, deltaY, image ):
    detectedPixels = 0
    obstaclePixels = deltaY*deltaX
    detectedPixelsInRect = 0 
    rows, columns = image.shape
    for row in range(rows):
        for column in range(columns):
            if(image[row][column]==255):
                detectedPixels = detectedPixels+1
    for row in range(minYCorner, minYCorner+deltaY):
        for column in range(minXCorner, minXCorner+deltaX):
            if(image[row][column]==255):
                detectedPixelsInRect = detectedPixelsInRect+1

    wrongDetection = abs(detectedPixels-detectedPixelsInRect)
    error_out_rect = wrongDetection/obstaclePixels*100
    error_in_rect = abs(detectedPixelsInRect-obstaclePixels)/obstaclePixels*100
    calcResults = {"detectedPixels":detectedPixels,
                   "detectedPixelsInRect":detectedPixelsInRect,
                   "obstaclePixels":obstaclePixels,
                   "error_out_rect" : error_out_rect,
                   "error_in_rect":error_in_rect}
    return calcResults

def temp():
    Right_nice = cv2.imread('/home/farshad/Desktop/latexFile_fristArticle_2ndSubmission_beforFirstRevise/result/end_dey_results_croped/1/2.png')
    Left_nice = cv2.imread('/home/farshad/Desktop/latexFile_fristArticle_2ndSubmission_beforFirstRevise/result/end_dey_results_croped/1/1.png')
    Left_nice_gray = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
    Right_nice_gray = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)


    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=45)
    disparity = stereo.compute(Left_nice_gray, Right_nice_gray)

    fig, ax = plt.subplots()
    print(type(disparity))
    ax.imshow(disparity,'gray')
    ax.set_title(
        "Click and drag to draw a rectangle.\n"
        "Press 't' to toggle the selector on and off.")

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # disable middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)

    plt.show()
