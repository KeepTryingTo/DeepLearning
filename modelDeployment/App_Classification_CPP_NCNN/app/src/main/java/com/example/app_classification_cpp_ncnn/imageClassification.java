package com.example.app_classification_cpp_ncnn;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
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
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Vector;


public class imageClassification extends AppCompatActivity implements View.OnClickListener {

    //加载本地库
    static {
        System.loadLibrary("app_classification_cpp_ncnn");
    }

    public List<String> classes;
    private Button btn_open;
    private Button btn_classify;
    private ImageView imageView;
    private Spinner spinner;
    private ActivityResultLauncher<Intent> register;
    private ArrayAdapter<CharSequence> adapter = null;
    private Uri uri;
    private static String[] model_name_list = {"custom_model","mobilenetv3"};
    private TextView displayResult;
    private Button btn_video;
    private Context context = this;

    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};
//    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
//    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    classifyUtils utls = new classifyUtils();

    AssetManager assetManager;
    public Bitmap bitmap;
    public int [] predictionIndex;
    public String modelName = "custom_model";
    public native void getModelName(String modelName);
    public native void loadModel(AssetManager assetManager);
    public native int[] DetectImage(Bitmap bitmap,float[] NO_MEAN_RGB,
                                              float[] NO_STD_RGB,int image_size,
                                              boolean is_use_gpu);

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
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
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
                    if(modelName.equals("custom_model")){
                        try {
                            classes = utls.readClassesFile(context,"class_custom.txt");
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        getModelName(modelName);
                        loadModel(getAssets());
                    }else if(modelName.equals("mobilenetv3")){
                        try {
                            classes = utls.readClassesFile(context,"imagenet_classes.txt");
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        getModelName(modelName);
                        loadModel(getAssets());
                    }
                }
                System.out.println("select model is : " + modelName);
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });
    }

    public Bitmap getBitmapFromUri(Context context, Uri uri) throws IOException {
        Bitmap image = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));
        return image;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.btn_open:
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                register.launch(intent);
                break;
            case R.id.btn_classify:
                if(uri != null){
                    try {
                        bitmap = getBitmapFromUri(this,uri);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
                //对图像缩放到指定的大小
                bitmap = classifyUtils.preprocessImage(bitmap,224,224,true);

                //开始对图像进行分类
                predictionIndex = DetectImage(bitmap,NO_MEAN_RGB,NO_STD_RGB,224,false);
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