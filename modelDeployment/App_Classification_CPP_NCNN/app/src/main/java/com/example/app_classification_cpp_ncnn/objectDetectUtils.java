package com.example.app_classification_cpp_ncnn;

import android.graphics.Rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
};

public class objectDetectUtils {
    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // model input image size
    static int mInputWidth = 640;
    static int mInputHeight = 640;

    // model output is of size 25200*(num_of_class+5)
    public static int mOutputRow = 25200; // as decided by the YOLOv5 model for input image of size 640*640
    public static int mOutputColumn = 85; // left, top, right, bottom, score and 80 class probability
    public static float mThreshold = 0.30f; // score above which a detection is generated
    public static int mNmsLimit = 15;

    static List<String> mClasses;

    // The two methods nonMaxSuppression and IOU below are ported from https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    static ArrayList<Result> nonMaxSuppression(ArrayList<Result> boxes, int limit, float iou_threshold) {

        // Do an argsort on the confidence scores, from high to low.对其所有的预测结果按照score进行从大到小的排序
        Collections.sort(boxes,
                new Comparator<Result>() {
                    @Override
                    public int compare(Result o1, Result o2) {
//                        return o1.score.compareTo(o2.score);//注意这个是原来代码，但是有问题，这里是从小到大排序，但是我们要的是从大到小排序
                        return o2.score.compareTo(o1.score);
                    }
                });

        ArrayList<Result> selected = new ArrayList<>();
        boolean[] active = new boolean[boxes.size()];//用于记录过滤之后的坐标框
        Arrays.fill(active, true);
        int numActive = active.length;

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        boolean done = false;
        for (int i=0; i<boxes.size() && !done; i++) {
            if (active[i]) {
                Result boxA = boxes.get(i);
                selected.add(boxA);
                if (selected.size() >= limit) break;

                for (int j=i+1; j<boxes.size(); j++) {
                    if (active[j]) {
                        Result boxB = boxes.get(j);
                        if (IOU(boxA.rect, boxB.rect) > iou_threshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }

    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    static float IOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = Math.min(a.right, b.right);
        float intersectionMaxY = Math.min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    static ArrayList<Result> outputsToNMSPredictions(float[][] outputs,int mOutputRow, float imgScaleX,
                                                     float imgScaleY, float ivScaleX,
                                                     float ivScaleY, float startX, float startY,
                                                     float conf_threshold,float iou_threshold) {
        ArrayList<Result> results = new ArrayList<>();
        for (float[] output : outputs) {
            if (output[4] > conf_threshold) {
                float x = output[0];
                float y = output[1];
                float w = output[2];
                float h = output[3];

                float left = imgScaleX * (x - w/2);
                float top = imgScaleY * (y - h/2);
                float right = imgScaleX * (x + w/2);
                float bottom = imgScaleY * (y + h/2);

                int cls = (int)output[5];

                Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY),
                                     (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
                //类别 + 置信度 + 矩形框
                Result result = new Result(cls, output[4], rect);
                results.add(result);
            }
        }
        return nonMaxSuppression(results, mNmsLimit, iou_threshold);
    }

    static ArrayList<Result> myOutputsToNMSPredictions(float[][] boxes,float[] scores,long[] labels
                                                     , float imgScaleX,
                                                     float imgScaleY, float ivScaleX,
                                                     float ivScaleY, float startX, float startY,
                                                     float conf_threshold,float iou_threshold) {
        ArrayList<Result> results = new ArrayList<>();
        System.out.println("object utils classes size: " + mClasses.size());
        mOutputRow = boxes.length;
        for (int i = 0; i< mOutputRow; i++) {
            if (scores[i] > conf_threshold) {
                float x1 = boxes[i][0];
                float y1 = boxes[i][1];
                float x2 = boxes[i][2];
                float y2 = boxes[i][3];
                //由于FCOS_ResNet50_FPN输出的坐标框是已经按照原图的大小比率进行了缩放，因此这里不需要*imgScaleX和*imgScaleY
                float left =imgScaleX * x1;
                float top = imgScaleY * y1;
                float right = imgScaleX * x2;
                float bottom = imgScaleY * y2;

                float score = scores[i];
                int cls = (int)labels[i];

                Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
                //类别 + 置信度 + 矩形框
                Result result = new Result(cls, score, rect);
                results.add(result);
            }
        }
        return nonMaxSuppression(results, mNmsLimit, iou_threshold);
    }
}
