
document.addEventListener('DOMContentLoaded', function() {
    const slider = document.getElementById('slider');
    const sliderValue = document.getElementById('slider-value');
    const submitButton = document.getElementById('submit-button');

    // 更新滑动条值
    slider.addEventListener('input', function() {
        sliderValue.textContent = slider.value / 100;
    });

    submitButton.addEventListener('click', function() {
                const value = sliderValue.textContent;
                alert(value);
                const formData = new FormData();
                formData.append('slider_value', value);
                fetch('/submit', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(data => {
                    alert(data);  // 可以在这里处理返回的数据
                });
            });
});
