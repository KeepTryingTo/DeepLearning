
document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const cameraSelect = document.getElementById('cameraSelect');

    async function getCameras() {
        /*
        调用异步方法：navigator.mediaDevices.enumerateDevices() 方法返回一个 Promise，
                    这个 Promise 在成功时解析为一个可用的媒体输入/输出设备列表（如摄像头、麦克风等）。
        等待结果：使用 await 关键字，代码会等待 enumerateDevices() 方法完成，并返回其结果。
                    这一过程是异步的，它不会阻塞主线程，其他的代码不会被阻塞执行。
        */
        const devices = await navigator.mediaDevices.enumerateDevices();
        //选择视频设备
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        //列出所有可用的视频设备到下拉菜单选项中
        videoDevices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${cameraSelect.length + 1}`;
            cameraSelect.appendChild(option);
        });
    }
    //初始化视频设备列表
    getCameras();

    async function startCamera() {
        //得到选择的视频设备
        const selectedCameraId = cameraSelect.value;
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: selectedCameraId ? { exact: selectedCameraId } : undefined }
        });
        //根据当前选择的视频设备设置流
        // video.srcObject = stream;
        video.src = "http://127.0.0.1:5000/video";
    }

    cameraSelect.addEventListener('change', startCamera);
});