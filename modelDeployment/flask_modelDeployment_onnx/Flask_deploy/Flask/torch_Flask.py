"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/9/14-9:09
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import time
import torch
import cvzone
import onnxruntime
import numpy as np
from PIL import Image
from torchvision.ops import nms
from werkzeug.utils import secure_filename
from Flask_deploy.detection import config
from Flask_deploy.detection.detection import loadModel,transform
from flask import Flask,render_template,request,redirect,url_for,abort,jsonify,Response


app=Flask(__name__)

conf_threshold = 0.20
iou_threshold = 0.45
model_name = "ssdlite320_mobilenet_v3"

img_size = 640
device = "cuda" if torch.cuda.is_available() else 'cpu'
className = []
with open(file=r'D:\conda3\Transfer_Learning\B Stand\day18\main\Flask_deploy\Flask\yolov5_classes.txt',
          mode='r', encoding='utf-8') as fp:
    lines = fp.readlines()
for line in lines:
    className.append(line.strip('\n'))
#加载模型
onnxYolov5 = onnxruntime.InferenceSession(r"D:\conda3\Transfer_Learning\B Stand\day18\main\Flask_deploy\weights\yolov5s.onnx")

@app.route('/selectModel',methods=['GET','POST'])
def selectModel():
    global model_name
    if request.method == 'POST':
        model_name = request.form.get('selected_model',type=str)
    return jsonify(model_name = model_name)

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def detectImage(threshold = 0.45,iou_threshold = 0.05,
                img_path = r'D:\conda3\Transfer_Learning\B站\day18\main\Flask_deploy\images\6.png'):
    startTime = time.time()
    imgName = os.path.basename(img_path)

    global model_name
    print("model name: {}".format(model_name))
    if model_name != "yolov5s":
        model = loadModel(model_name = model_name)
        image = Image.open(img_path)
        #TODO PIL convert OpenCV(便于后面将框绘制到图像上)
        cv_img = np.array(image)
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        height,width,_ = cv_img.shape

        image = transform(image).unsqueeze(dim=0).to(config.device)
        outs = model(image)
        #TODO NMS丢弃那些重叠的框，如果一个置信度最大的框和其他框之间的IoU > iou_threshold，那么就表示重叠并且需要丢弃
        indexs = nms(boxes=outs[0]['boxes'],scores=outs[0]['scores'],iou_threshold=iou_threshold)

        boxes = outs[0]['boxes'][indexs]
        scores = outs[0]['scores'][indexs]
        labels = outs[0]['labels'][indexs]

        print('boxes.shape: {}'.format(outs[0]['boxes'][indexs].shape))
        print('scores.shape: {}'.format(outs[0]['scores'][indexs].shape))
        print('labels.shape: {}'.format(outs[0]['labels'][indexs].shape))

        for i in range(boxes.size()[0]):
            box = boxes[i]
            confidence = scores[i]
            label = labels[i]
            if confidence > threshold:
                box = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]

                cv2.rectangle(
                    img=cv_img,pt1=(box[0],box[1]),pt2=(box[2],box[3]),
                    color=(255, 0, 255),thickness=1
                )

                text = "{} {}%".format(config.className[int(label.item())],round(confidence.item() * 100,2))
                cvzone.putTextRect(
                    img=cv_img,text=text,pos=(box[0] + 9,box[1] - 12),scale=0.5,thickness=1,colorR=(0,255,0),
                    font=cv2.FONT_HERSHEY_SIMPLEX
                )
    elif model_name == "yolov5s":
        im0 = Image.open(img_path)
        width, height = im0.size

        im = im0.resize(size=(img_size, img_size))
        im = transform(im).unsqueeze(dim=0).to('cpu')
        im = im.numpy()

        predictions = onnxYolov5.run(output_names=['output0'], input_feed={'images': im})
        # [1,25200,85] = [1,25200,xywh + conf + 80]
        print('predictions.shape: {}'.format(predictions[0].shape))
        pred = np.squeeze(predictions)
        boxes = xywh2xyxy(pred[..., :4])
        confidences = pred[..., 4]
        cls_prob = pred[..., 5:85]
        labels = np.argmax(cls_prob, axis=-1)
        confidences = confidences * np.max(cls_prob, axis=-1)
        boxes, confidences, labels = (torch.tensor(boxes, dtype=torch.float32),
                                      torch.tensor(confidences, dtype=torch.float32), torch.tensor(labels))

        indexs = nms(boxes=boxes, scores=confidences, iou_threshold=iou_threshold)
        boxes = boxes[indexs]
        confidences = confidences[indexs]
        labels = labels[indexs]

        # TODO PIL convert Opencv
        cv_img = np.array(im0)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        for k in range(boxes.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(boxes[k][0] / img_size * width)
            yleft = int(boxes[k][1] / img_size * height)
            xright = int(boxes[k][2] / img_size * width)
            yright = int(boxes[k][3] / img_size * height)

            confidence = confidences[k].item()
            class_id = labels[k].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_threshold:
                text = className[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(cv_img, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=cv_img, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
    endTime =  time.time()
    runTime = endTime - startTime
    print('detect finished {} time is: {}s'.format(imgName, runTime))

    return cv_img,runTime

def timeDetect(threshold = 0.3):
    global model_name
    model = loadModel(model_name = model_name)
    # 计算开始时间
    start_time = time.time()
    # 计算帧率
    countFPS = 0
    # 开启摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(src=frame, dsize=(config.crop_size, config.crop_size))
        frame = cv2.flip(src=frame, flipCode=2)
        # 将其opencv读取的图像格式转换为PIL读取的类型格式
        frame_PIL = Image.fromarray(frame)
        img_transform = transform(frame_PIL)
        # 对图像进行升维
        img_Transform = torch.unsqueeze(input=img_transform, dim=0).to(config.device)

        detection = model(img_Transform)
        index = nms(boxes = detection[0]['boxes'],scores=detection[0]['scores'],iou_threshold=iou_threshold)
        boxes = detection[0]['boxes'][index]
        labels = detection[0]['labels'][index]
        scores = detection[0]['scores'][index]

        # 获取类别概率值
        end_time = time.time()
        countFPS += 1
        FPS = round(countFPS / (end_time - start_time), 0)
        cv2.putText(img=frame, text='FPS: ' + str(FPS), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 0), thickness=2)

        for k in range(len(labels)):
            xleft = int(boxes[k][0])
            yleft = int(boxes[k][1])
            xright = int(boxes[k][2])
            yright = int(boxes[k][3])

            class_id = labels[k].item()
            confidence = scores[k].item()

            if confidence > threshold:
                text = config.className[class_id] + ': ' + str('{:.2f}%'.format(round(confidence * 100,2)))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        image = cv2.imencode('.jpg', frame)[1].tobytes()
        ## 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route(rule='/submit_conf',methods=['POST','GET'])
def submit_conf():
    global conf_threshold
    conf_threshold = request.form.get('slider_value', type = float)
    print("conf_threshold: ",conf_threshold)
    return jsonify(value = conf_threshold)

@app.route(rule='/submit_iou',methods=['POST','GET'])
def submit_iou():
    global iou_threshold
    iou_threshold = request.form.get('slider_iou_value', type = float)
    print("iou_threshold: ",iou_threshold)
    return jsonify(value = iou_threshold)

@app.route(rule='/image',methods=['POST','GET'])
def appDetectImage():
    global conf_threshold
    global iou_threshold
    if request.method=='POST':
        file = request.files.get('filename')
        if file:
            #对文件进行完全检测
            filename=secure_filename(file.filename)
            print('filename: {}'.format(file.filename))
            # 获得当前的图片
            save_imgPath=os.getcwd()+'\\static\\images\\'+filename
            print('save image path: {}'.format(save_imgPath))
            file.save(save_imgPath)

            cv_img,runTime = detectImage(threshold=conf_threshold,iou_threshold=iou_threshold,img_path=save_imgPath)
            #保存当前的图片到static文件夹中
            if os.path.isfile(save_imgPath):
                os.remove(save_imgPath)
                cv2.imwrite(os.path.join(os.getcwd(),'static','images',filename),cv_img)
            # cv2.imshow('img',cv_img)
            # cv2.waitKey(0)

            response={
                "detect_time": round(runTime * 1000,2),
                "image_path":"./static/images/"+filename
            }
            #注意这里返回到index中的图片路径问题
            return render_template(template_name_or_list='detectImage.html',response=response)
    response={
        "detect_time": "",
        'image_path':""
    }
    return render_template(template_name_or_list='detectImage.html',response=response)

@app.route(rule='/video', methods=['POST', 'GET'])
def appVideo():
    ## 这个地址返回视频流响应
    return Response(timeDetect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route(rule='/',methods=['POST','GET'])
def index():
    if request.method == "POST":
        btn_type = request.form.get('submit')
        if btn_type == "detectImage":
            return redirect(url_for("image"))
        elif btn_type == "detectTime":
            return redirect(url_for("video"))
    return render_template('index.html')


if __name__ == '__main__':
    print('Pycharm')
    app.run(debug=True)