import sys
import time

import cv2
import numpy as np


def build_model(is_cuda):
    net = cv2.dnn.readNet("best.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

# cam_url='https://172.27.54.87:4343/video'
cam_url = './test.mp4'


def load_capture():
    # capture = cv2.VideoCapture("test.mp4")
    capture = cv2.VideoCapture(cam_url)
    return capture

capture = load_capture()

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_detection = cv2.VideoWriter('detection1.avi', fourcc , 20, (frame_width, frame_height),True)  # 保存视频
out_carline = cv2.VideoWriter('carline1.avi', fourcc , 20, (frame_width, frame_height),True)  # 保存视频




def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def get_center(box):
    center_x = box[0]+box[2]/2
    center_y = box[1]+box[3]/2
    return (center_x,center_y)

"""标定获参 getParameters.py"""
camera_matrix0 = [[1.67428973e+03, 0.00000000e+00, 3.70472988e+02],
 [0.00000000e+00, 2.07399431e+03, 6.63332179e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist_coeffs = [[-4.19798197e-01, -5.16270488e+01,  5.97212799e-02 ,-1.87775294e-01,4.47226134e+02]]
rvecs = np.array((np.array([[-1.86431124],
       [-1.92693405],
       [-0.41884511]]),))
tvecs = np.array((np.array([[-48.44594595],
       [-38.20562424],
       [227.5616861 ]]),))

"""像素坐标系-->世界坐标系"""
def pixel2world(image_point_center_x,image_point_center_y):
    x = [image_point_center_x,image_point_center_y,1]
    image_point = np.array(x).reshape(3, 1)
    image_point = np.asmatrix(image_point)

    rotate_matrix = cv2.Rodrigues(rvecs[0])[0]
    translate_matrix = tvecs[0]

    rotate_matrix = np.asmatrix(rotate_matrix)
    translate_matrix = np.asmatrix(translate_matrix)
    camera_matrix = np.asmatrix(camera_matrix0)

    camera_rotate = camera_matrix * rotate_matrix
    camera_translate = camera_matrix * translate_matrix
    camera_rotate_inv = np.linalg.inv(camera_rotate)

    world_point_z = 0
    matrix1 = camera_rotate_inv * camera_translate

    matrix2 = camera_rotate_inv * image_point

    world_point_x = ((matrix1[2][0] + world_point_z) * matrix2[0] / matrix2[2]) - matrix1[0]
    world_point_y = ((matrix1[2][0] + world_point_z) * matrix2[1] / matrix2[2]) - matrix1[1]
    # print(worldPoint[worldPoint_index])
    # print([world_point_x[0], world_point_y[0], 0])
    world_point_x = np.array(world_point_x)
    world_point_y = np.array(world_point_y)
    # return (world_point_x[0][0], world_point_y[0][0])
    return (world_point_y[0][0], world_point_x[0][0])



class_list = ['center','front']
colors = [(255, 255, 0), (0, 255, 255)]

is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

net = build_model(is_cuda)
# capture = load_capture()

start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1

def nothing(*arg):
    pass

icol = (23, 87, 46, 77, 255, 255) ## 绿色
# Lower range colour sliders.
cv2.namedWindow('colorTest',)
# cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
# cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
# cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# # Higher range colour sliders.
# cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
# cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
# cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)



while (capture.isOpened()):

    _, frame = capture.read()
    frame_carline = frame

    if frame is None:
        print("End of stream")
        break

    

    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    frame_count += 1
    total_frames += 1

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
         color = colors[int(classid) % len(colors)]
         cv2.rectangle(frame, box, color, 2)
         cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
         cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0))
        
         if class_list[classid] == 'center':
            center_x,center_y = get_center(box)
            world_point_center_x, world_point_center_y = pixel2world(center_x,center_y)
            world_point_center_x = round(world_point_center_x,1)
            world_point_center_y = round(world_point_center_y,1)
            cv2.putText(frame, 'coords:'+'('+str(world_point_center_x)+','+str(world_point_center_y)+')', (10,int(frame_height/4)-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    
    # 画箭头
    cv2.arrowedLine(frame, (10, int(frame_height/4)), (10, int(frame_height/4)+70), (29, 67, 147), 2, 9, 0, 0.3)
    cv2.putText(frame,'y',(20, int(frame_height/4)+80),cv2.FONT_HERSHEY_COMPLEX,1,(29, 67, 147),2)
    cv2.arrowedLine(frame, (10, int(frame_height/4)), (80, int(frame_height/4)), (29, 67, 147), 2, 9, 0, 0.3)
    cv2.putText(frame, 'x', (90, int(frame_height/4)), cv2.FONT_HERSHEY_COMPLEX, 1, (29, 67, 147), 2)

    cv2.circle(frame, (10, int(frame_height/4)), 15, (0, 0, 255), -1)
    out_detection.write(frame)

    # """车道"""
    # lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    # lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    # lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    # highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    # highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    # highVal = cv2.getTrackbarPos('highVal', 'colorTest')
    lowHue = icol[0]
    lowSat = icol[1]
    lowVal = icol[2]
    highHue = icol[3]
    highSat = icol[4]
    highVal = icol[5]

    frameBGR = cv2.GaussianBlur(frame_carline, (7, 7), 0)
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    result_color = cv2.bitwise_and(frame_carline, frame_carline, mask=mask)
    cv2.imshow('colorTest', result_color)
    out_carline.write(result_color)


    if frame_count >= 30:
        end = time.time_ns()
        fps = 1000000000 * frame_count / (end - start)
        frame_count = 0
        start = time.time_ns()
    
    # if fps > 0:
    #     fps_label = "FPS: %.2f" % fps
    #     cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.namedWindow("output", 0)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break


capture.release()
out.release()
cv2.destroyAllWindows()

print("Total frames: " + str(total_frames))
