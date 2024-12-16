from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
from algorithms.naive_bayes import naive_bayes_classifier
from algorithms.utils import update_dataset_info
from algorithms.kmeans import preprocess_data, kmeans_clustering
from algorithms.apriori import apriori_algorithm
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
        
        # Đọc file và lấy thông tin dataset
        df = pd.read_excel(filepath)
        dataset_info = {
            'columns': list(df.columns),
            'unique_values': {col: df[col].unique().tolist() for col in df.columns}
        }
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
        "apriori":"apriori.html"
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
        uploaded_file = file.filename
        file.save(f'./uploads/{uploaded_file}')

        # Read the dataset and store the column information
        dataset_info = pd.read_excel(f'./uploads/{uploaded_file}')
        
        # Convert the dataset columns to a list for use in the template
        dataset_info = {
            'columns': dataset_info.columns.tolist(),
            'unique_values': {col: dataset_info[col].dropna().unique().tolist() for col in dataset_info.columns}
        }
        
        # Redirect back to the same algorithm page to show updated dataset info
        return redirect(url_for('show_algorithm', name='naive_bayes'))

# Kết nối logic của naive bayes
@app.route('/run_naive_bayes', methods=['POST'])
def run_naive_bayes():
    global dataset_info, uploaded_file
    target_column = request.form['target_column']
    selected_columns = request.form.getlist('selected_columns')
    input_values = {col: request.form[f'input_values[{col}]'] for col in selected_columns}
    
        # Lấy giá trị từ checkbox "use_laplace"
    use_laplace = request.form.get('use_laplace') == 'true'  # Nếu checkbox được chọn, giá trị sẽ là 'true'

    # Đọc dataset từ file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    df = pd.read_excel(filepath)

    # Gọi logic Naive Bayes
    from algorithms.naive_bayes import naive_bayes_classifier
    result = naive_bayes_classifier(df, target_column, input_values, use_laplace)

    # Trả về giao diện với kết quả
    result_message = f"Dữ liệu thuộc lớp: {result}"  # Hoặc tùy chỉnh theo cách bạn muốn hiển thị kết quả

    # Chọn template phù hợp để trả về
    return render_template('naive_bayes.html', 
                           dataset_info=dataset_info, 
                           result=result_message, uploaded_file = uploaded_file)

# Kết nối logic của k-means
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    result = None
    cluster_filter = None

    # Lấy số cụm K từ form
    k = int(request.form.get('k'))
    cluster_filter = request.form.get('cluster_filter')

    # Lấy dataset đã được upload
    uploaded_file = session.get('uploaded_file', None)

    if uploaded_file:
        # Đọc file từ đường dẫn đã được tải lên
        data = pd.read_csv(uploaded_file)

        # Tiền xử lý dữ liệu
        preprocessed_data = preprocess_data(data)

        # Chạy K-Means
        clustered_data, centroids = kmeans_clustering(preprocessed_data, k)

        # Lọc theo cụm nếu người dùng chọn
        if cluster_filter:
            filtered_data = clustered_data[clustered_data['Cluster'] == int(cluster_filter)]
        else:
            filtered_data = clustered_data

        result = {
            'data': filtered_data.to_html(classes='table table-striped', index=False),
            'centroids': centroids
        }

    return render_template(
        'kmeans.html',
        result=result,
        k=request.form.get('k', None),
        cluster_filter=cluster_filter,
        uploaded_file=uploaded_file
    )
@app.route('/run_apriori', methods=['POST'])
def run_apriori():
    global dataset_info, uploaded_file

    # Lấy các tham số từ form
    transaction_columns = request.form.getlist('transaction_columns')
    min_support = float(request.form['min_support'])
    min_confidence = float(request.form['min_confidence'])

    # Đọc dataset từ file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    df = pd.read_excel(filepath)

    # Chuyển đổi dữ liệu thành danh sách giao dịch
    transactions = df[transaction_columns].values.tolist()

    # Gọi thuật toán Apriori
    result = apriori_algorithm(transactions, min_support, min_confidence)

    # Trả về kết quả cho giao diện
    return render_template('apriori.html',
                           dataset_info=dataset_info,
                           result=result,
                           uploaded_file=uploaded_file)
if __name__ == '__main__':
    # Đảm bảo thư mục upload tồn tại
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)



# Để thêm thuật toán mới, chỉ cần tạo file Python trong 
# algorithms/ và thêm logic xử lý vào app.py và algorithm.html.
# Có thể thêm tính năng trực quan hóa kết quả (biểu đồ, bảng) vào giao diện từng thuật toán.
