
document.addEventListener('DOMContentLoaded', function() {
    const slider_conf = document.getElementById('slider_conf');
    const slider_iou = document.getElementById('slider_iou');

    const sliderValue = document.getElementById('slider_conf_value');
    const sliderIOUValue = document.getElementById('slider_iou_value');

    const submitButton = document.getElementById('submit_confidence_button');
    const submitIOUButton = document.getElementById('submit_iou_button');

    // 更新置信度滑动条值
    slider_conf.addEventListener('input', function() {
        sliderValue.textContent = slider_conf.value / 100;
    });
    // 更新IOU滑动条值
    slider_iou.addEventListener('input', function() {
        sliderIOUValue.textContent = slider_iou.value / 100;
    });

    submitButton.addEventListener('click', function() {
                const value = sliderValue.textContent;//获取当前拖动滑动条改变的值
                const formData = new FormData();
                formData.append('slider_value', value);
                fetch('/submit_conf', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(data => {
                    alert(data);  // 可以在这里处理返回的数据
                });
            });
    submitIOUButton.addEventListener('click', function() {
                const value = sliderIOUValue.textContent;//获取当前拖动滑动条改变的值
                const formData = new FormData();
                formData.append('slider_iou_value', value);
                fetch('/submit_iou', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(data => {
                    alert(data);  // 可以在这里处理返回的数据
                });
            });
});
