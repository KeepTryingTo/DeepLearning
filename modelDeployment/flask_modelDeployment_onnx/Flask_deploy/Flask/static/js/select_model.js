
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('submit_modelName_button').addEventListener('click', function() {
            const selectedValue = document.getElementById('dropdown').value;
            // 创建一个 FormData 对象
            const formData = new FormData();
            formData.append('selected_model', selectedValue);

            // 发送数据到 Flask 后端
            fetch('/selectModel', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(data => {
                alert('服务器响应: ' + data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
});
