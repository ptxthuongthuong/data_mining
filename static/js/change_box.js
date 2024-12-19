function updateTargetValues() {
    // Lấy giá trị của combobox "Chọn thuộc tính quyết định"
    var targetColumn = document.getElementById('target_column').value;
  
    // Kiểm tra nếu có giá trị được chọn
    if (targetColumn) {
      // Lấy các giá trị duy nhất từ cột tương ứng trong dataset_info
      var uniqueValues = dataset_info[targetColumn];
  
      // Lấy combobox "Chọn giá trị của tập mục tiêu"
      var targetValueSelect = document.getElementById('target_value');
  
      // Xóa tất cả các tùy chọn hiện tại
      targetValueSelect.innerHTML = '';
  
      // Thêm các tùy chọn mới vào combobox
      uniqueValues.forEach(function(value) {
        var option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        targetValueSelect.appendChild(option);
      });
    }
  }
  