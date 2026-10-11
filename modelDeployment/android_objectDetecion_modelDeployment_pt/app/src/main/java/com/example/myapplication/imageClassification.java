package com.example.myapplication;

import android.annotation.SuppressLint;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import android.net.Uri;
import android.os.Bundle;

import android.provider.MediaStore;

import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageProxy;
import androidx.core.app.ActivityCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

//import org.pytorch.Tensor;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;

import java.util.List;
import java.util.Map;


public class imageClassification extends AppCompatActivity implements View.OnClickListener {

    public List<String> classes;
    private Button btn_open;
    private Button btn_classify;
    private ImageView imageView;
    private Spinner spinner;
    private ActivityResultLauncher<Intent> register;
    private ArrayAdapter<CharSequence> adapter = null;
    private String modelName = "custom_model";
    private Uri uri;
    private static String[] model_name_list = {"custom_model","mobilenetv3"};
    private TextView displayResult;
    private Button btn_video;
    private Context context = this;
    private Module model;

    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};
//    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
//    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    classifyUtils utls = new classifyUtils();

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_image_classification);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        btn_open = findViewById(R.id.btn_open);
        btn_classify = findViewById(R.id.btn_classify);
        imageView = findViewById(R.id.imageView);
        spinner = findViewById(R.id.spinner);
        displayResult = findViewById(R.id.dislayResult);
        btn_video = findViewById(R.id.btn_video);
        //最开始设置图像
        try {
            String filePath = assertFilePath(this,"cat.png");
            File file = new File(filePath);
            uri = Uri.fromFile(file);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            imageView.setImageBitmap(bitmap);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //监听按钮
        btn_open.setOnClickListener(this);
        btn_classify.setOnClickListener(this);
        btn_video.setOnClickListener(this);

        register = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult o) {
                Intent intent = o.getData();
                if (intent != null){
                    uri = intent.getData();
                    imageView.setImageURI(intent.getData());
                }
            }
        });

        //字符串给到适配器中
        adapter = new ArrayAdapter<CharSequence>(this,
                android.R.layout.simple_spinner_dropdown_item,model_name_list);
        spinner.setAdapter(adapter);

        //获取下拉菜单中的值
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                modelName = adapterView.getItemAtPosition(i).toString();

                //根据选择的模型来加载当前模型和类别文件
                if(modelName.length() > 0){
                    System.out.println("select model is : " + modelName);
                    //读取类别文件
                    //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
                    try {
                        if(modelName.equals("custom_model")){
                            classes = utls.readClassesFile(context,"class_custom.txt");
                            model = utls.loadModel(context,"custom_model.pt",false);
                        }else if(modelName.equals("mobilenetv3")){
                            classes = utls.readClassesFile(context,"imagenet_classes.txt");
                            model = utls.loadModel(context,"mobilenet_v3_small.pt",false);
                        }
                        System.out.println("classes size: " + classes.size());
//                       for(String className : classes){
//                           System.out.println("className: " + className);
//                       }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });
    }

    public Bitmap getBitmapFromUri(Context context, Uri uri) throws IOException {
        //context.getContentResolver()：获取应用的 content resolver，它允许访问和操作设备上的数据（如文件、数据库、内容提供者等）。
        //ParcelFileDescriptor 对象，该对象表示文件的底层文件描述符
//        ParcelFileDescriptor parcelFileDescriptor =
//                context.getContentResolver().openFileDescriptor(uri, "r");
//
//        if (parcelFileDescriptor == null) throw new FileNotFoundException("Failed to load image.");
////        从 ParcelFileDescriptor 对象中获取文件描述符。FileDescriptor 是一个指向打开文件的句柄，可以与文件相关的操作一起使用。
//        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
////        使用文件描述符解码图像并将其转换为 Bitmap 对象。这个方法能够处理不同格式的图像（如 JPEG、PNG等）
//        Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);
//        parcelFileDescriptor.close();
        Bitmap image = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));
        return image;
    }

    //根据预测的结果返回最大类别概率的索引
    public int[] predict(Tensor inputTensor){
        //从输出中提取数据,输出shape = [1,1000]
        final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
//        getDataAsFloatArray() 方法将 outputTensor 中的数据转换为一个一维 float 数组
        float [] pred = outputTensor.getDataAsFloatArray();
        float [] result = null;
        if(modelName.equals("mobilenetv3")){
             result = utls.softMax(pred);
        }else{
            result = pred;
        }
        int[] predictionIndex = utls.getTopk(result,3);
        return predictionIndex;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.btn_open:
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                register.launch(intent);
                break;
            case R.id.btn_classify:
                //根据加载的模型和选择的图像进行前向推理
                Bitmap bitmap = null;
                try {
                    bitmap = getBitmapFromUri(this,uri);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                Tensor imgProcess = utls.preprocessImage(bitmap,NO_MEAN_RGB,NO_STD_RGB,
                        224,224,true);
                int[] predictionIndex = predict(imgProcess);
                String predictName = "";
                for(int i = 0 ; i < predictionIndex.length; i++){
                    if(i == predictionIndex.length - 1){
                        predictName += classes.get(predictionIndex[i]);
                    }else{
                        predictName += classes.get(predictionIndex[i]) + "\n";
                    }
                }
                displayResult.setText(predictName);
                break;
            case R.id.btn_video:
                Intent intentVideo = new Intent(imageClassification.this, classificationVideo.class);
                intentVideo.putExtra("modelName",modelName);
                System.out.println("video start");
//                register.launch(intentVideo);
                startActivity(intentVideo);
                break;
            default:
                break;
        }
    }

    //获取图像路径
    public static String assertFilePath(Context context, String assertName) throws IOException {
        File file = new File(context.getFilesDir(),assertName);
        if(file.exists() && file.length() > 0){
            return file.getAbsolutePath();
        }
        try {
            InputStream inputStream = context.getAssets().open(assertName);
            OutputStream outputStream = new FileOutputStream(file);
            byte [] buffer = new byte[4 * 1024];
            int read;
            while((read = inputStream.read(buffer)) != -1){
                outputStream.write(buffer,0,read);
            }
            outputStream.flush();
        }catch (IOException e){
            e.printStackTrace();
        }
        return file.getAbsolutePath();
    }


    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {
        super.onPointerCaptureChanged(hasCapture);
    }
}