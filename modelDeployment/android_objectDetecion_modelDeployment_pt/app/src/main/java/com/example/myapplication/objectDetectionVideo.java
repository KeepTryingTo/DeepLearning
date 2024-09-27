package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class objectDetectionVideo extends AbstractCameraXActivity<objectDetectionVideo.AnalysisResult> {

    private float[][] boxes;
    private float[] scores;
    private long[] labels;
    private float[][] outputs;
    private int mOutputRow = 100;
    private int img_size = 320;
    private List<String> mClasses;
    private String modelName;

    private Float conf_threshold;
    private Float iou_threshold;

    private Module model;

    private objectResultView mResultView;
    objectDetectUtils det_utils = new objectDetectUtils();
    classifyUtils cls_utils = new classifyUtils();

    private Context context = this;

    private void configuration(){
        modelName = getIntent().getStringExtra("modelName");
        img_size = getIntent().getIntExtra("imgSize",0);
        conf_threshold = getIntent().getFloatExtra("confThreshold",0);
        iou_threshold = getIntent().getFloatExtra("iouThreshold",0);
        //根据选择的模型来加载当前模型和类别文件
        if(modelName.length() > 0){
            System.out.println("video select model is : " + modelName);
            //读取类别文件
            //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
            try {
                if(modelName.equals("ssdlite320_mobilenet_v3_large")){
                    img_size = 320;
                    mClasses = cls_utils.readClassesFile(context, "pytorch_classes.txt");
                    model = cls_utils.loadModel(context,"ssdlite320_mobilenet_v3_large.pt",true);
                }else if(modelName.equals("yolov5s")){
                    img_size = 640;
                    mOutputRow = 25200;
                    mClasses = cls_utils.readClassesFile(context, "yolov5_classes.txt");
                    model = cls_utils.loadModel(context,"yolov5s.torchscript.pt",true);
                }
                System.out.println("video classes size: " + mClasses.size());
                objectDetectUtils.mClasses = mClasses;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection_video;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.detResultView);
//        当调用 inflate() 方法时，ViewStub 会将其引用的布局文件填充到视图层次结构中。
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    //根据预测的结果返回最大类别概率的索引
    public void predict(Tensor inputTensor) {
        // 创建输入格式特征和输出格式结果 [1,25200,85]
        final IValue[] outputTensor = model.forward(IValue.from(inputTensor)).toTuple();

//        // 获取 outputTensor 的维度和大小
        final Tensor output = outputTensor[0].toTensor();
        long[] sizes = output.shape();
        int numDimensions = sizes.length;
//
        float[][][] outputData = new float[(int) sizes[0]][(int) sizes[1]][(int) sizes[2]];

        float[] flatData = output.getDataAsFloatArray();
        int index = 0;

        for (int i = 0; i < sizes[0]; i++) {
            if (numDimensions == 3) {
                outputData[i] = new float[(int) sizes[1]][(int) sizes[2]];
                for (int j = 0; j < sizes[1]; j++) {
                    for (int k = 0; k < sizes[2]; k++) {
                        outputData[i][j][k] = flatData[index++];
                    }
                }
            }
        }

        if(modelName.equals("yolov5s")){
            outputs = outputData[0];
        }else if(modelName.equals("ssdlite320_mobilenet_v3_large")){

        }
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        configuration();

        Bitmap bitmap = imgToBitmap(image.getImage());

        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        float imgScaleX = (float)bitmap.getWidth() / img_size;
        float imgScaleY = (float)bitmap.getHeight() / img_size;
        float ivScaleX = (float)mResultView.getWidth() / bitmap.getWidth();
        float ivScaleY = (float)mResultView.getHeight() / bitmap.getHeight();

        ArrayList<Result> results;
        Tensor imgProcess = cls_utils.preprocessImage(bitmap,objectDetectUtils.NO_MEAN_RGB,
                                            objectDetectUtils.NO_STD_RGB,img_size,img_size,true);
        predict(imgProcess);

        if(modelName.equals("ssdlite320_mobilenet_v3_large")){
            results = det_utils.myOutputsToNMSPredictions(boxes,scores,labels,
                    imgScaleX, imgScaleY,ivScaleX, ivScaleY, 0, 0,conf_threshold,iou_threshold);
        }else if(modelName.equals("yolov5s")){
            results =  det_utils.outputsToNMSPredictions(outputs,mOutputRow, imgScaleX, imgScaleY,ivScaleX,
                    ivScaleY, 0, 0,conf_threshold,iou_threshold);
        } else {
            results = null;
        }
        return new AnalysisResult(results);
    }
}