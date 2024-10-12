package com.example.translate_transformer;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONException;
import org.json.JSONObject;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;

public class translate_page extends AppCompatActivity implements View.OnClickListener,Runnable {

    static {
        System.loadLibrary("translate_transformer");
    }

    private EditText fran_edit;
    private EditText en_edit;
    private Button btn_translate;
    private Spinner spinner;
    private Context context = this;

    private String modelName = "seq2seq";
    private static String[] model_name_list = {"seq2seq","transformer"};
    private ArrayAdapter<CharSequence> adapter = null;
    private JSONObject word2idx;
    private JSONObject idx2word;
    private long[] inputs;

    private static final int HIDDEN_SIZE = 256;
    private static final int EOS_TOKEN = 1;
    private static final int MAX_LENGTH = 50;
    private static final String TAG = MainActivity.class.getName();

    private Module mModuleEncoder;
    private Module mModuleDecoder;
    private Tensor mInputTensor;
    private LongBuffer mInputTensorBuffer;

    @SuppressLint({"ResourceAsColor", "MissingInflatedId", "ResourceType"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_translate_page);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        fran_edit = findViewById(R.id.fran);
        en_edit = findViewById(R.id.en);
        btn_translate = findViewById(R.id.btn_translate);
        spinner = findViewById(R.id.spinner);

        GradientDrawable drawable_en = new GradientDrawable();
        drawable_en.setStroke(4, R.color.teal_200); // 边框宽度和颜色
        drawable_en.setCornerRadius(8f); // 圆角半径
        fran_edit.setBackground(drawable_en);

        GradientDrawable drawable = new GradientDrawable();
        drawable.setStroke(4, R.color.teal_200); // 边框宽度和颜色
        drawable.setCornerRadius(8f); // 圆角半径
        en_edit.setBackground(drawable);

        //给输入文本框设置监听事件
        fran_edit.setOnClickListener(this);
        en_edit.setOnClickListener(this);
        btn_translate.setOnClickListener(this);

        //字符串给到适配器中
        adapter = new ArrayAdapter<CharSequence>(this,
                android.R.layout.simple_spinner_dropdown_item,model_name_list);
        spinner.setAdapter(adapter);
//        adapter.setDropDownViewResource(R.id.spinner_font_size);
//        adapter.setDropDownViewResource(R.style.CustomSpinnerItemStyle);

        //获取下拉菜单中的值
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                modelName = adapterView.getItemAtPosition(i).toString();
                //根据选择的模型来加载当前模型和类别文件
                if(modelName.length() > 0){
                    if(modelName.equals("seq2seq")){
                        word2idx = loadJSONFromAsset(context,"source_wrd2idx.json");
                        idx2word = loadJSONFromAsset(context,"target_idx2wrd.json");
                        //TODO 加载解码器模型
                        mModuleDecoder=loadModel(mModuleDecoder,"optimized_decoder_150k.ptl");
                        //TODO 加载编码器模型
                        mModuleEncoder=loadModel(mModuleEncoder,"optimized_encoder_150k.ptl");
                    }else if(modelName.equals("transformer")){
                        word2idx = loadJSONFromAsset(context,"source_word2idx.json");
                        idx2word = loadJSONFromAsset(context,"target_idx2word.json");
                        //TODO 加载解码器模型
                        mModuleDecoder=loadModel(mModuleDecoder,"optimized_decoder_150k.ptl");
                        //TODO 加载编码器模型
                        mModuleEncoder=loadModel(mModuleEncoder,"optimized_encoder_150k.ptl");
                    }
                }
                System.out.println("select model is : " + modelName);
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });
    }

    private JSONObject loadJSONFromAsset(Context context, String fileName) {
        String json = null;
        JSONObject word2idx = null;
        AssetManager assetManager = context.getAssets();

        try (InputStream is = assetManager.open(fileName);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {

            StringBuilder stringBuilder = new StringBuilder();
            String line;

            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line);
            }
            json = stringBuilder.toString();
            word2idx = new JSONObject(json);
            System.out.println("word2idx: " + word2idx.length());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }

        return word2idx;
    }

    public Module loadModel(Module model,String modelName){
        if (model == null) {
            try {
                model = LiteModuleLoader.load(assetFilePath(getApplicationContext(), modelName));
            } catch (IOException e) {
                Log.e(TAG, "Error reading assets", e);
                finish();
            }
        }
        return model;
    }

    public String predict(){
        //TODO 定义输入和隐藏向量的形状
        final long[] inputShape = new long[]{1};
        final long[] hiddenShape = new long[]{1, 1, 256};
        final FloatBuffer hiddenTensorBuffer =
                Tensor.allocateFloatBuffer(1 * 1 * 256);
        Tensor hiddenTensor = Tensor.fromBlob(hiddenTensorBuffer, hiddenShape);

        final long[] outputsShape = new long[]{MAX_LENGTH, HIDDEN_SIZE};
        final FloatBuffer outputsTensorBuffer =
                Tensor.allocateFloatBuffer(MAX_LENGTH  * HIDDEN_SIZE);

        for (int i=0; i<inputs.length; i++) {
            LongBuffer inputTensorBuffer = Tensor.allocateLongBuffer(1);
            inputTensorBuffer.put(inputs[i]);
            Tensor inputTensor = Tensor.fromBlob(inputTensorBuffer, inputShape);
            final IValue[] outputTuple = mModuleEncoder.forward(IValue.from(inputTensor), IValue.from(hiddenTensor)).toTuple();
            final Tensor outputTensor = outputTuple[0].toTensor();
            outputsTensorBuffer.put(outputTensor.getDataAsFloatArray());
            hiddenTensor = outputTuple[1].toTensor();
        }

        Tensor outputsTensor = Tensor.fromBlob(outputsTensorBuffer, outputsShape);
        final long[] decoderInputShape = new long[]{1, 1};


        mInputTensorBuffer = Tensor.allocateLongBuffer(1);
        mInputTensorBuffer.put(0);
        mInputTensor = Tensor.fromBlob(mInputTensorBuffer, decoderInputShape);
        ArrayList<Integer> result = new ArrayList<>(MAX_LENGTH);
        for (int i=0; i<MAX_LENGTH; i++) {
            final IValue[] outputTuple = mModuleDecoder.forward(
                    IValue.from(mInputTensor),
                    IValue.from(hiddenTensor),
                    IValue.from(outputsTensor)).toTuple();
            final Tensor decoderOutputTensor = outputTuple[0].toTensor();
            hiddenTensor = outputTuple[1].toTensor();
            float[] outputs = decoderOutputTensor.getDataAsFloatArray();
            int topIdx = 0;
            double topVal = -Double.MAX_VALUE;
            for (int j=0; j<outputs.length; j++) {
                if (outputs[j] > topVal) {
                    topVal = outputs[j];
                    topIdx = j;
                }
            }

            if (topIdx == EOS_TOKEN) break;

            result.add(topIdx);
            mInputTensorBuffer = Tensor.allocateLongBuffer(1);
            mInputTensorBuffer.put(topIdx);
            mInputTensor = Tensor.fromBlob(mInputTensorBuffer, decoderInputShape);
        }

        String english = "";
        for (int i = 0; i < result.size(); i++) {
            try {
                english += " " + idx2word.getString("" + result.get(i));
            } catch (JSONException e) {
                throw new RuntimeException(e);
            }
        }
        return english;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.fran:
                fran_edit.setText("");
                break;
            case R.id.en:
                en_edit.setText("");
                break;
            case R.id.btn_translate:
                //TODO 获得输入框的法语句子
                String french = fran_edit.getText().toString();
                //TODO 判断当前是否输入了要翻译的句子
                if(french.length() == 0){
                    Toast.makeText(this,"请输入要翻译的句子",Toast.LENGTH_SHORT).show();
                    return;
                }
                //然后根据英文的空格划分单词
                inputs =  new long [french.split(" ").length];
                String []french_list = french.split(" ");
                int size = french_list.length;
                try {
                    for (int i = 0; i < size; i++) {
                        inputs[i] = word2idx.getLong(french_list[i]);
                    }
                }
                catch (JSONException e) {
                    android.util.Log.e(TAG, "JSONException ", e);
                }

                Thread thread = new Thread(translate_page.this);
                thread.start();;
                break;
        }
    }

    public void run() {
        final String result = predict();
        runOnUiThread(() -> {
            en_edit.setText(result);
            btn_translate.setEnabled(true);
        });
    }
    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}