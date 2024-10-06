package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.ContentHandler;
import java.nio.FloatBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class classifyUtils {
    public OrtEnvironment env = null;
    public OrtSession session = null;
    public static int img_size = 224;

    //加载ONNX模型
    public void OnnxModel(Context context, String modelName) throws OrtException {
        // 初始化ONNX Runtime环境
        env = OrtEnvironment.getEnvironment();

        // 从assets中获取模型文件的输入流
        InputStream inputStream = null;
        try {
            inputStream = context.getAssets().open(modelName);
            // 在流中创建临时文件以被加载
            File tempModelFile = File.createTempFile("model", "onnx");
            //使用完临时文件后，应当显式地删除它以释放系统资源;deleteOnExit()方法会在JVM退出时删除文件
            tempModelFile.deleteOnExit();
            FileOutputStream out = new FileOutputStream(tempModelFile);

            byte[] buffer = new byte[1024];
            int length;
            //将输出流中读取的内容写入到到临时文件tempModelFile中
            while ((length = inputStream.read(buffer)) != -1) {
                out.write(buffer, 0, length);
            }
            out.close();
            // 加载模型
            session = env.createSession(tempModelFile.getAbsolutePath(), new OrtSession.SessionOptions());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public List<List<Integer>> readClassesFile(Context context, String fileName) throws IOException {
        List<List<Integer>> platte = new ArrayList<>();
        try {
            String line;
            InputStream inputStream = context.getAssets().open(fileName);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));

            while((line = bufferedReader.readLine()) != null){
                List<Integer> RGB = new ArrayList<>();
                Integer R = Integer.parseInt(line.split(",")[0].trim());
                Integer G = Integer.parseInt(line.split(",")[1].trim());
                Integer B = Integer.parseInt(line.split(",")[2].trim());
                RGB.add(R);
                RGB.add(G);
                RGB.add(B);
                platte.add(RGB);
            }
            bufferedReader.close();
            inputStream.close();
        }catch (IOException e){
            e.printStackTrace();
        }
        return platte;
    }

    public float[] softMax(float [] predictions){
        float[] probs = new float[predictions.length];
        float maxLogit = Float.NEGATIVE_INFINITY; //得到系统的负无穷

        //遍历找到predictions中的最大值，避免溢出
        for(float logit : predictions){
            if(maxLogit < logit){
                maxLogit = logit;
            }
        }
        float sum = 0.0f;
        //根据softmax计算公式
        for(int i = 0 ; i < predictions.length; i++){
            probs[i] = (float) Math.exp(predictions[i] - maxLogit);
            sum += probs[i];
        }
        for(int i = 0; i < probs.length; i++){
            probs[i] /= sum;
        }
        return probs;
    }
    //获取前N个最大概率类别
    public int[] getTopk(float [] predictions,int N){
        float[] topProbabilities = new float[N];
        int[] topIndices = new int[N];
        //遍历类别数
        for (int i = 0; i < predictions.length; i++) {
            //遍历Top-k的插入排序
            for (int j = 0; j < N; j++) {
                if (predictions[i] > topProbabilities[j]) {
                    // 插入新的概率
                    for (int k = N - 1; k > j; k--) {
                        topProbabilities[k] = topProbabilities[k - 1];
                        topIndices[k] = topIndices[k - 1];
                    }
                    topProbabilities[j] = predictions[i];
                    topIndices[j] = i;
                    break;
                }
            }
        }
        // 输出 Top-N 类别及其概率
        for (int i = 0; i < N; i++) {
            System.out.println("Top " + (i + 1) + " Class Index: " + topIndices[i] + ", Probability: " + topProbabilities[i]);
        }
        return topIndices;
    }

    public int[][] getMask(float[][][][]prediction){
        // 获取每一维的大小
        int N = prediction.length;
        int C = prediction[0].length;
        int H = prediction[0][0].length;
        int W = prediction[0][0][0].length;

        int [][] mask = new int[H][W];
        for(int n = 0 ; n < N; n ++){
            for(int i = 0; i < H; i++){
                for(int j = 0 ; j < W; j++){
                    float maxvalue = 0.0f;
                    int classId = 0;
                    for(int c = 0 ; c < C; c++){
                        if(maxvalue < prediction[n][c][i][j]){
                            maxvalue = prediction[n][c][i][j];
                            classId = c;
                        }
                    }
                    mask[i][j] = classId;
                }
            }
        }
        return mask;
    }

    public Bitmap getPlatteBitmap(int[][]mask,Bitmap bitmap,List<List<Integer>>platte){
        int W = bitmap.getWidth();
        int H = bitmap.getHeight();
        //创建一个三通道的bitmap
        Bitmap platteBitmap = Bitmap.createBitmap(W,H, Bitmap.Config.ARGB_8888);
        for(int i = 0 ; i < H; i++){
            for(int j = 0 ; j < W; j++){
                //获取当前预测类别的颜色
                List<Integer> color = platte.get(mask[i][j]);
                int org_color = bitmap.getPixel(i,j);
                int alpha = Color.alpha(org_color);
                int R = color.get(0);
                int G = color.get(1);
                int B = color.get(2);
                platteBitmap.setPixel(j,i,Color.argb(alpha,R,G,B));
            }
        }
        return platteBitmap;
    }

    private static void checkOutBufferCapacity(
            FloatBuffer outBuffer, long outBufferOffset, int tensorWidth, int tensorHeight) {
        if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
            throw new IllegalStateException("Buffer underflow");
        }
    }

    private static void checkNormStdArg(float[] normStdRGB) {
        if (normStdRGB.length != 3) {
            throw new IllegalArgumentException("normStdRGB length must be 3");
        }
    }

    private static void checkNormMeanArg(float[] normMeanRGB) {
        if (normMeanRGB.length != 3) {
            throw new IllegalArgumentException("normMeanRGB length must be 3");
        }
    }

    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset,
            final boolean isContinus) {
        checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);
        checkNormMeanArg(normMeanRGB);
        checkNormStdArg(normStdRGB);

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, width, x, y, width, height);
        if (isContinus) {
            final int offset_g = pixelsCount;
            final int offset_b = 2 * pixelsCount;
            for (int i = 0; i < pixelsCount; i++) {
                final int c = pixels[i];
                float r = ((c >> 16) & 0xff) / 255.0f;
                float g = ((c >> 8) & 0xff) / 255.0f;
                float b = ((c) & 0xff) / 255.0f;
                outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
                outBuffer.put(outBufferOffset + offset_g + i, (g - normMeanRGB[1]) / normStdRGB[1]);
                outBuffer.put(outBufferOffset + offset_b + i, (b - normMeanRGB[2]) / normStdRGB[2]);
            }
        } else {
            for (int i = 0; i < pixelsCount; i++) {
                final int c = pixels[i];
                float r = ((c >> 16) & 0xff) / 255.0f;
                float g = ((c >> 8) & 0xff) / 255.0f;
                float b = ((c) & 0xff) / 255.0f;
                outBuffer.put(outBufferOffset + 3 * i + 0, (r - normMeanRGB[0]) / normStdRGB[0]);
                outBuffer.put(outBufferOffset + 3 * i + 1, (g - normMeanRGB[1]) / normStdRGB[1]);
                outBuffer.put(outBufferOffset + 3 * i + 2, (b - normMeanRGB[2]) / normStdRGB[2]);
            }
        }
    }

    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {
        bitmapToFloatBuffer(
                bitmap,
                x,
                y,
                width,
                height,
                normMeanRGB,
                normStdRGB,
                outBuffer,
                outBufferOffset,
                true);
    }

    //根据选择的图像进行处理
    public static FloatBuffer preprocessImage(Bitmap bitmap,float[] NO_MEAN_RGB,float[] NO_STD_RGB,
                                            int img_size_w,int img_size_h,boolean isContinus) {
        Bitmap resizedBitmap = null;
            // 根据模型输入大小对图像进行缩放;第四个参数 true：指示是否使用双线性滤波来进行缩放。开启后能获得更好的图片质量
        resizedBitmap = Bitmap.createScaledBitmap(bitmap, img_size_w, img_size_h, true);
        System.out.println("resize mBitmap width: " + (float)resizedBitmap.getWidth());
        System.out.println("resize mBitmap height: " + (float)resizedBitmap.getHeight());
//        //根据缩放之后图像大小申请存储像素空间
//        int[] pixels = new int[resizedBitmap.getWidth() * resizedBitmap.getHeight()];
//        //设置其遍历像素的开始位置以及步长;并将位图resizedBitmap中的像素存储在pixels；
//        // 第四个参数 0：原始图像的起始行，表示从顶部开始。
//        //第五个参数 0：原始图像的起始列，表示从左侧开始。
//        resizedBitmap.getPixels(pixels, 0, resizedBitmap.getWidth(), 0, 0,
//                resizedBitmap.getWidth(), resizedBitmap.getHeight());
//        //由于是RGB彩色图像，因此申请的空间也需要根据实际来
//        float [] floatValues = new float[resizedBitmap.getWidth() * resizedBitmap.getHeight() * 3];
//        for (int i = 0; i < pixels.length; i++) {
//            // 传入浮点值 范围 [0, 1]，还可以根据模型需求做归一化
//            //使用位运算>>和&来分别提取红色、绿色和蓝色的值。由于每个颜色通道占用8位（即0-255的范围），
//            // 因此通过右移16位（对于红色）、8位（对于绿色）和0位（对于蓝色），并使用& 0xFF来保留最低8位，
//            // 可以分别获取到这三个颜色通道的值。
//            final int c = pixels[i];
//            float red = ((c >> 16) & 0xFF) / 255.0f;
//            float green = ((c >> 8) & 0xFF) / 255.0f;
//            float blue = (c & 0xFF) / 255.0f;
//            // 归一化处理
//            floatValues[i * 3 + 0] = (red - NO_MEAN_RGB[0]) / NO_STD_RGB[0];   // R
//            floatValues[i * 3 + 1] = (green - NO_MEAN_RGB[1]) / NO_STD_RGB[1]; // G
//            floatValues[i * 3 + 2] = (blue - NO_MEAN_RGB[2]) / NO_STD_RGB[2];  // B
//        }
//        return floatValues;
        final FloatBuffer floatBuffer = FloatBuffer.allocate(3 * img_size_h * img_size_w);
        bitmapToFloatBuffer(
                resizedBitmap,0,0,
                img_size_w,img_size_h,
                NO_MEAN_RGB,NO_STD_RGB,
                floatBuffer,
                0,
                isContinus);
        return floatBuffer;
    }
}
