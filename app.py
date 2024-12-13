from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
from algorithms.naive_bayes import naive_bayes_classifier
from algorithms.utils import update_dataset_info
from algorithms.kmeans import preprocess_data, kmeans_clustering

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Biến toàn cục để lưu thông tin dataset
uploaded_file = None
dataset_info = None

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html', 
                           file_uploaded=uploaded_file, 
                           dataset_info=dataset_info if dataset_info else {})

# Xử lý upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file, dataset_info
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        uploaded_file = file.filename
        
        # Gọi hàm xử lý dataset từ utils.py
        from utils import update_dataset_info
        dataset_info = update_dataset_info(filepath)
        
        return redirect(url_for('index'))
    return "File upload failed!", 400

# Giao diện thuật toán
@app.route('/algorithm/<name>')
def show_algorithm(name):
    global dataset_info

    # Xác định template cho thuật toán
    templates = {
        "naive_bayes": "naive_bayes.html",
        "decision_tree": "decision_tree.html",
        "kmeans" : "kmeans.html",
        # Thêm các thuật toán khác tại đây
    }

    if name not in templates:
        return "Thuật toán không tồn tại", 404

    # Trả về giao diện tương ứng
    return render_template(templates[name], algo_name=name, dataset_info=dataset_info, uploaded_file = uploaded_file)

dataset_info = {}  # To store information about the dataset
uploaded_file = ""  # To store the filename of the uploaded file

@app.route('/upload_new_dataset', methods=['POST'])
def upload_new_dataset():
    global dataset_info, uploaded_file

    if 'new_dataset_file' not in request.files:
        return redirect(request.url)

    file = request.files['new_dataset_file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filepath = os.path.join('./uploads', file.filename)
        file.save(filepath)
        uploaded_file = file.filename

        # Gọi hàm xử lý dataset từ utils.py
        from utils import update_dataset_info
        dataset_info = update_dataset_info(filepath)

        # Lấy thuật toán hiện tại từ form
        current_algorithm = request.form.get('current_algorithm', 'naive_bayes')

        # Redirect back to the current algorithm page
        return redirect(url_for('show_algorithm', name=current_algorithm))

# Kết nối logic của naive bayes
@app.route('/run_naive_bayes', methods=['POST'])
def run_naive_bayes():
    try:
        global dataset_info, uploaded_file
        
        # Kiểm tra dữ liệu đầu vào
        if not uploaded_file or not dataset_info:
            return "Vui lòng tải lên file dữ liệu trước", 400

        target_column = request.form.get('target_column')
        selected_columns = [col.strip() for col in request.form.getlist('selected_columns')]
        
        if not target_column or not selected_columns:
            return "Thiếu thông tin cần thiết", 400

        # Lấy input values cho các cột được chọn
        input_values = {col: request.form.get(f'input_values[{col}]') for col in selected_columns}
        use_laplace = request.form.get('use_laplace') == 'true'

        # Đọc dataset
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        if not os.path.exists(filepath):
            return "File dữ liệu không tồn tại", 400
            
        df = pd.read_excel(filepath)

        # Gọi naive bayes classifier
        from algorithms.naive_bayes import naive_bayes_classifier
        result = naive_bayes_classifier(df, target_column, input_values, use_laplace)

        # Tạo message kết quả
        result_message = f"Dữ liệu thuộc lớp: {result}"

        # Kiểm tra nếu là AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # Trả về HTML kết quả trực tiếp
            return f'''
                <h3>Kết quả phân lớp:</h3>
                <p>{result_message}</p>
            '''
        
        # Nếu không phải AJAX, trả về full page
        return render_template('naive_bayes.html',
                             dataset_info=dataset_info,
                             result=result_message,
                             uploaded_file=uploaded_file)

    except Exception as e:
        error_message = f"Có lỗi xảy ra: {str(e)}"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return error_message, 500
        return render_template('naive_bayes.html',
                             dataset_info=dataset_info,
                             result=error_message,
                             uploaded_file=uploaded_file)

# Kết nối logic của k-means
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    try:
        global dataset_info, uploaded_file
       
        # Kiểm tra dữ liệu đầu vào
        if not uploaded_file or not dataset_info:
            return "Vui lòng tải lên file dữ liệu trước", 400
           
        k = request.form.get('k')
        if not k or not k.isdigit():
            return "Số cụm K không hợp lệ", 400
        k = int(k)
        
        # Thêm print để kiểm tra dữ liệu
        print("Raw selected_columns:", request.form.getlist('selected_columns'))
        
        # Loại bỏ ngoặc vuông nếu có
        selected_columns = [col.strip('[]"') for col in request.form.getlist('selected_columns')]
        
        # Loại bỏ các khoảng trắng thừa
        selected_columns = [col.strip() for col in selected_columns]
        
        print("Processed selected_columns:", selected_columns)
        
        if not selected_columns:
            return "Vui lòng chọn ít nhất một cột", 400
       
        # Đọc file dữ liệu
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        if not os.path.exists(filepath):
            return "File dữ liệu không tồn tại", 400
           
        # Đọc file dựa vào định dạng
        file_extension = uploaded_file.split('.')[-1].lower()
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        elif file_extension == 'csv':
            df = pd.read_csv(filepath)
        else:
            return "Định dạng file không được hỗ trợ", 400
           
        # Kiểm tra các cột được chọn có tồn tại trong DataFrame
        invalid_columns = [col for col in selected_columns if col not in df.columns]
        if invalid_columns:
            return f"Các cột sau không tồn tại: {', '.join(invalid_columns)}", 400
        
        # Giới hạn dữ liệu theo các cột được chọn
        df = df[selected_columns]
            
        # Kiểm tra dữ liệu trống
        if df.empty:
            return "File dữ liệu trống", 400
            
        # Kiểm tra K không được lớn hơn số mẫu
        if k > len(df):
            return f"Số cụm K ({k}) không được lớn hơn số mẫu dữ liệu ({len(df)})", 400
            
        # Xử lý missing values
        df = df.dropna()
        if df.empty:
            return "Dữ liệu không hợp lệ sau khi xử lý missing values", 400
            
        # Thực hiện clustering
        try:
            preprocessed_data = preprocess_data(df)
            clustered_data, centroids = kmeans_clustering(preprocessed_data, k)
        except Exception as e:
            app.logger.error(f"Lỗi trong quá trình clustering: {str(e)}")
            return f"Lỗi khi thực hiện clustering: {str(e)}", 500
            
        # Tạo nội dung kết quả
        result_html = f"""
            <hr>
            <h3>Kết quả</h3>
            <h4>Thông tin dataset:</h4>
            <p>Số mẫu dữ liệu: {len(df)}</p>
            <p>Số thuộc tính: {len(df.columns)}</p>
            <h4>Centroids:</h4>
            <pre>{str(centroids)}</pre>
            <h4>Bảng phân cụm:</h4>
            <div>{clustered_data.to_html(classes='table table-striped', index=False)}</div>
        """
        
        # Kiểm tra loại request và trả về kết quả phù hợp
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return result_html
        return render_template('kmeans.html',
                            dataset_info=dataset_info,
                            result=result_html,
                            uploaded_file=uploaded_file)
                            
    except Exception as e:
        app.logger.error(f"Lỗi không mong đợi: {str(e)}")
        error_message = f"Có lỗi xảy ra: {str(e)}"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return error_message, 500
        return render_template('kmeans.html',
                            dataset_info=dataset_info,
                            result=error_message,
                            uploaded_file=uploaded_file)
if __name__ == '__main__':
    # Đảm bảo thư mục upload tồn tại
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)



# Để thêm thuật toán mới, chỉ cần tạo file Python trong 
# algorithms/ và thêm logic xử lý vào app.py và algorithm.html.
# Có thể thêm tính năng trực quan hóa kết quả (biểu đồ, bảng) vào giao diện từng thuật toán.