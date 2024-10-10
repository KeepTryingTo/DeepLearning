package com.example.app_classification_cpp_ncnn;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class classifyUtils {

    public static int img_size = 224;


    public List<String> readClassesFile(Context context, String fileName) throws IOException {
        List<String> classes = new ArrayList<>();
        try {
            String line;
            InputStream inputStream = context.getAssets().open(fileName);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));

            while((line = bufferedReader.readLine()) != null){
                classes.add(line.trim());
            }
            bufferedReader.close();
            inputStream.close();
        }catch (IOException e){
            e.printStackTrace();
        }
        return classes;
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


    //根据选择的图像进行处理
    public static Bitmap preprocessImage(Bitmap bitmap,
                                            int img_size_w,int img_size_h,boolean isContinus) {
        Bitmap resizedBitmap = null;
            // 根据模型输入大小对图像进行缩放;第四个参数 true：指示是否使用双线性滤波来进行缩放。开启后能获得更好的图片质量
        resizedBitmap = Bitmap.createScaledBitmap(bitmap, img_size_w, img_size_h, true);
        System.out.println("resize mBitmap width: " + (float)resizedBitmap.getWidth());
        System.out.println("resize mBitmap height: " + (float)resizedBitmap.getHeight());
        return resizedBitmap;
    }
}
