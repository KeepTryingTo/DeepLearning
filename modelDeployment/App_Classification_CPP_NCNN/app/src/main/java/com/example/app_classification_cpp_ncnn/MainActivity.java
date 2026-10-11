package com.example.app_classification_cpp_ncnn;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.example.app_classification_cpp_ncnn.databinding.ActivityMainBinding;

import java.util.Random;

public class MainActivity extends AppCompatActivity implements RadioGroup.OnCheckedChangeListener, View.OnClickListener  {

    // Used to load the 'app_classification_cpp_ncnn' library on application startup.
    //加载本地库
    static {
        System.loadLibrary("app_classification_cpp_ncnn");
    }

    private TextView tv_password;
    private EditText et_password;
    private Button btn_forget;
    private CheckBox ck_remember;
    private EditText et_phone;
    private RadioButton rb_password;
    private RadioButton rb_verifycode;
    private ActivityResultLauncher<Intent> register;
    private Button btn_login;
    private String mPassword = "121212";
    private String mVerifyCode;
    private SharedPreferences preferences;
    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        RadioGroup rg_login = findViewById(R.id.rg_login);
        tv_password = findViewById(R.id.tv_password);
        et_password = findViewById(R.id.et_password);
        btn_forget = findViewById(R.id.btn_forget);
        ck_remember = findViewById(R.id.ck_remember);
        et_phone = findViewById(R.id.et_phone);
        rb_password = findViewById(R.id.rb_password);
        rb_verifycode = findViewById(R.id.rb_verifycode);
        btn_login = findViewById(R.id.btn_login);
        //设置单选监听事件
        rg_login.setOnCheckedChangeListener(this);
        //给输入号码和输入密码文本编辑框添加监听事件
        et_phone.addTextChangedListener(new HideTextWatcher(et_phone,11));
        et_password.addTextChangedListener(new HideTextWatcher(et_password,6));
        //给“忘记密码”按钮添加监听事件
        btn_forget.setOnClickListener(this);
        btn_login.setOnClickListener(this);

        register = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult o) {
                Intent intent = o.getData();
                if(intent != null && o.getResultCode() == MainActivity.RESULT_OK){
                    mPassword = intent.getStringExtra("new_password");
                }
            }
        });

        //记住密码
        preferences = getSharedPreferences("config", Context.MODE_PRIVATE);
        reload();
    }


    private void reload() {
        boolean isRemember = preferences.getBoolean("isRemember",false);
        if(isRemember){
            String phone = preferences.getString("phone","");
            et_phone.setText(phone);

            String password = preferences.getString("password","");
            et_password.setText(password);

            ck_remember.setChecked(true);
        }
    }

    @Override
    public void onCheckedChanged(RadioGroup group,int checkedId){
        Log.d("RadioGroup","checkedId: " + checkedId);
        switch (checkedId){
//            选择密码登录
            case R.id.rb_password:
                tv_password.setText(getString(R.string.login_password));
                et_password.setHint(getString(R.string.input_password));
                btn_forget.setText(getString(R.string.forget_password));
                ck_remember.setVisibility(TextView.VISIBLE);
                break;
//            选择验证码登录方式
            case R.id.rb_verifycode:
                tv_password.setText(getString(R.string.verifycode));
                et_password.setHint(getString(R.string.input_verifycode));
                btn_forget.setText(getString(R.string.get_verifycode));
                ck_remember.setVisibility(TextView.GONE);
                break;
            default:
                break;
        }
    }

    @Override
    public void onClick(View view) {
        String phone = et_phone.getText().toString();
        if(phone.length() < 11){
            Toast.makeText(this,"请输入正确的号码",Toast.LENGTH_SHORT).show();
            return;
        }
        switch (view.getId()){
            case R.id.btn_forget:

                //表示当前选择密码的登录方式
                if(rb_password.isChecked()){
                    //跳转到忘记密码的界面
                    Intent intent = new Intent(this,LoginForgetActivity.class);
                    intent.putExtra("phone",phone);
                    register.launch(intent);
                }else if(rb_verifycode.isChecked()){
                    //生成随机的6为验证码
                    mVerifyCode = String.format("%06d",new Random().nextInt(999999));
                    //弹出一个窗口，提示用户记住验证码数字
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setTitle("请记住验证码");
                    builder.setMessage("手机号" + phone + ",本次验证码是" + mVerifyCode + ",请输入验证码");
                    builder.setPositiveButton("好的",null);
                    AlertDialog dialog = builder.create();
                    dialog.show();
                }
                break;
            case R.id.btn_login:
                //如果是密码登录方式，判断输入的密码是否正确
                if(rb_password.isChecked()){
                    if(!mPassword.equals(et_password.getText().toString())){
                        Toast.makeText(this,"请输入正确的密码",Toast.LENGTH_SHORT).show();
                        return;
                    }

                    loginSuccess();
                }else if(rb_verifycode.isChecked()){
                    if(!mVerifyCode.equals(et_password.getText().toString())){
                        Toast.makeText(this,"请输入正确的密码",Toast.LENGTH_SHORT).show();
                        return;
                    }
                    //提示登录成功
                    loginSuccess();
                }

                break;
        }
    }

    private void loginSuccess(){
        String desc = String.format("您的手机号码是%s,恭喜您通过验证,点击确定按钮返回上个页面",et_phone.getText().toString());
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("登录成功");
        builder.setMessage(desc);
        builder.setPositiveButton("确定返回", (dialogInterface, which) -> {
            //结束当前的页面
            finish();
        });
        builder.setNegativeButton("我再看看",null);
        AlertDialog dialog = builder.create();
        dialog.show();

        if(ck_remember.isChecked()){
            SharedPreferences.Editor editor = preferences.edit();
            editor.putString("phone",et_phone.getText().toString());
            editor.putString("password",et_password.getText().toString());
            editor.putBoolean("isRemember",ck_remember.isChecked());
            editor.commit();
        }
        //提示登录成功
        System.out.println("login successfully!");
        //选择图像分类和目标检测的
        Intent intent = new Intent(this, ImageDetection.class);
        register.launch(intent);
    }

    //定义一个编辑框监听器，在输入文本达到指定长度时自动隐藏输入法
    public class HideTextWatcher implements TextWatcher {
        private EditText mView;
        private int mMaxLength;
        public HideTextWatcher(EditText et, int maxLength) {
            this.mView = et;
            this.mMaxLength = maxLength;
        }

        @Override
        public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

        }

        @Override
        public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

        }

        @Override
        public void afterTextChanged(Editable editable) {
            if(editable.toString().length() == mMaxLength){
                //输入法给隐藏
                ViewUtil.hideOneInputMethod(MainActivity.this,mView);
            }

        }
    }
}