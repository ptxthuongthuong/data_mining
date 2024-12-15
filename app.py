from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
from algorithms.naive_bayes import naive_bayes_classifier
from algorithms.utils import update_dataset_info, encode_data
from algorithms.kmeans import preprocess_data, kmeans_clustering
from algorithms.desision_tree import ID3DecisionTree
from sklearn.tree import plot_tree
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
        dataset_info = update_dataset_info(filepath)

        # Lấy thuật toán hiện tại từ form
        current_algorithm = request.form.get('current_algorithm', 'naive_bayes')

        # Redirect back to the current algorithm page
        return redirect(url_for('show_algorithm', name=current_algorithm))

# Kết nối logic của naive bayes
from flask import render_template_string

from flask import Flask, request, render_template
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/run_naive_bayes', methods=['POST'])
def run_naive_bayes():
    try:
        global dataset_info, uploaded_file
        
        # Kiểm tra dữ liệu đầu vào
        if not uploaded_file or not dataset_info:
            logger.error('No file uploaded or dataset info missing')
            return "Vui lòng tải lên file dữ liệu trước", 400

        # Lấy thông tin từ form
        target_column = request.form.get('target_column')
        selected_columns = [col.strip() for col in request.form.getlist('selected_columns')]
        
        if not target_column or not selected_columns:
            logger.error('Missing target column or selected columns')
            return "Thiếu thông tin cần thiết", 400
        
        # Lấy giá trị input cho mỗi cột được chọn
        input_values = {col: request.form.get(f'input_values[{col}]') for col in selected_columns}
        
        # Kiểm tra Laplace smoothing
        use_laplace = request.form.get('use_laplace') == 'true'
        
        # Kiểm tra file tồn tại
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        if not os.path.exists(filepath):
            logger.error(f'File not found: {filepath}')
            return "File dữ liệu không tồn tại", 400
        
        # Đọc dữ liệu
        df = pd.read_excel(filepath)
        logger.debug(f'Data loaded successfully. Shape: {df.shape}')
        
        # Thực hiện phân lớp
        try:
            result, priors, likelihoods, posteriors = naive_bayes_classifier(
                df, target_column, input_values, use_laplace
            )
            logger.debug('Classification completed successfully')
        except Exception as e:
            logger.error(f'Classification error: {str(e)}', exc_info=True)
            return f"Lỗi khi thực hiện phân lớp: {str(e)}", 500
        
        # Tạo bảng HTML cho kết quả
        priors_table = '<table border="1"><tr><th>Class</th><th>Prior Probability</th></tr>'
        for target_class, prior in priors.items():
            priors_table += f'<tr><td>{target_class}</td><td>{prior:.4f}</td></tr>'
        priors_table += '</table>'

        # Likelihoods Table
        likelihoods_table = '<table border="1"><tr><th>Feature</th><th>Class</th><th>Likelihood</th></tr>'
        for feature, class_likelihoods in likelihoods.items():
            for target_class, likelihood in class_likelihoods.items():
                likelihoods_table += f'<tr><td>{feature}</td><td>{target_class}</td><td>{likelihood:.4f}</td></tr>'
        likelihoods_table += '</table>'

        # Posteriors Table
        posteriors_table = '<table border="1"><tr><th>Class</th><th>Posterior Probability</th></tr>'
        for target_class, posterior in posteriors.items():
            posteriors_table += f'<tr><td>{target_class}</td><td>{posterior:.4f}</td></tr>'
        posteriors_table += '</table>'
        
        result_message = f"<strong>Dữ liệu thuộc lớp: {result}</strong>"
        
        # Nếu là AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return f'''
                <div class="result-section">
                    <h3>Kết quả phân lớp:</h3>
                    <p class="result-message">{result_message}</p>
                    <div class="calculation-details">
                        <h4>Chi tiết tính toán:</h4>
                        <div class="table-section">
                            <h5>Priors (Xác suất tiên nghiệm):</h5>
                            {priors_table}
                        </div>
                        <div class="table-section">
                            <h5>Likelihoods (Xác suất có điều kiện):</h5>
                            {likelihoods_table}
                        </div>
                        <div class="table-section">
                            <h5>Posteriors (Xác suất hậu nghiệm):</h5>
                            {posteriors_table}
                        </div>
                    </div>
                </div>
            '''
        
        # Nếu không phải AJAX request
        return render_template(
            'naive_bayes.html',
            result=result_message,
            priors_table=priors_table,
            likelihoods_table=likelihoods_table,
            posteriors_table=posteriors_table
        )
        
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}', exc_info=True)
        return "Có lỗi xảy ra khi xử lý yêu cầu", 500

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

        selected_columns = [col.strip('[]"').strip() for col in request.form.getlist('selected_columns')]

        if not selected_columns:
            return "Vui lòng chọn ít nhất một cột", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        if not os.path.exists(filepath):
            return "File dữ liệu không tồn tại", 400

        file_extension = uploaded_file.split('.')[-1].lower()
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        elif file_extension == 'csv':
            df = pd.read_csv(filepath)
        else:
            return "Định dạng file không được hỗ trợ", 400

        invalid_columns = [col for col in selected_columns if col not in df.columns]
        if invalid_columns:
            return f"Các cột sau không tồn tại: {', '.join(invalid_columns)}", 400

        df = df[selected_columns]

        if df.empty:
            return "File dữ liệu trống", 400

        if k > len(df):
            return f"Số cụm K ({k}) không được lớn hơn số mẫu dữ liệu ({len(df)})", 400

        df = df.dropna()
        if df.empty:
            return "Dữ liệu không hợp lệ sau khi xử lý missing values", 400

        try:
            preprocessed_data = preprocess_data(df)
            clustered_data, centroids = kmeans_clustering(preprocessed_data, k)
        except Exception as e:
            app.logger.error(f"Lỗi trong quá trình clustering: {str(e)}")
            return f"Lỗi khi thực hiện clustering: {str(e)}", 500

        # Format centroids rõ ràng hơn
        centroids_html = "<br>".join([
            f"Cụm {i + 1}: {', '.join([f'{value:.4f}' for value in centroid])}"
            for i, centroid in enumerate(centroids)
        ])

        # Đổi tên cột Cluster thành "Cụm"
        clustered_data.rename(columns={"Cluster": "Cụm"}, inplace=True)

        # Tạo HTML bảng kết quả với tùy chọn lọc theo cụm
        unique_clusters = clustered_data['Cụm'].unique()
        filter_html = "<label for='filter_cluster'>Chọn cụm:</label>"
        filter_html += "<select id='filter_cluster' onchange='filterCluster()'>"
        filter_html += "<option value=''>Tất cả</option>"
        for cluster in unique_clusters:
            filter_html += f"<option value='{cluster}'>Cụm {cluster}</option>"
        filter_html += "</select>"

        table_html = clustered_data.to_html(classes='table table-striped', index=False)

        result_html = f"""
            <hr>
            <h3>Kết quả</h3>
            <h4>Thông tin dataset:</h4>
            <p>Số mẫu dữ liệu: {len(df)}</p>
            <p>Số thuộc tính: {len(df.columns)}</p>
            <h4>Centroids:</h4>
            <p>{centroids_html}</p>
            <h4>Bảng phân cụm:</h4>
            <div>
                {filter_html}
                <div id='cluster_table'>
                    {table_html}
                </div>
            </div>
            <script>
                function filterCluster() {{
                    const selectedCluster = document.getElementById('filter_cluster').value;
                    const rows = document.querySelectorAll('#cluster_table table tbody tr');
                    rows.forEach(row => {{
                        const clusterCell = row.cells[row.cells.length - 1];
                        if (selectedCluster === '' || clusterCell.textContent === selectedCluster) {{
                            row.style.display = '';
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }});
                }}
            </script>  
        """

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

# Kết nối logic của Decsition_tree
@app.route('/run_decision_tree', methods=['POST'])
def run_decision_tree():
    try:
        global dataset_info, uploaded_file

        # Kiểm tra file đã upload hay chưa
        if not uploaded_file or not dataset_info:
            return {"error": "Vui lòng tải lên dataset trước khi chạy thuật toán"}, 400

        # Đọc file dataset
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        df = pd.read_excel(filepath)

        # Lấy thông tin từ form
        criterion = request.form.get('criterion', 'entropy')  # Mặc định là 'entropy'
        target_column = request.form.get('target_column')
        selected_columns = request.form.getlist('selected_columns')

        if not target_column or not selected_columns:
            return {"error": "Vui lòng chọn cột target và các cột dữ liệu"}, 400

        # Chuẩn bị dữ liệu
        X = df[selected_columns]
        y = df[target_column]

        # Kiểm tra và chuyển đổi dữ liệu target nếu cần
        from sklearn.preprocessing import LabelEncoder
        le_y = LabelEncoder()

        # Nếu target là dữ liệu liên tục, chuyển thành phân loại
        if pd.api.types.is_numeric_dtype(y):
            # Thử chia thành các nhóm (bins) nếu là dữ liệu liên tục
            try:
                y_encoded = le_y.fit_transform(pd.qcut(y, q=5, labels=False))
            except ValueError:
                # Nếu không thể chia bins, sử dụng phương pháp khác
                y_encoded = le_y.fit_transform(pd.cut(y, bins=5, labels=False))
        else:
            # Nếu đã là dữ liệu phân loại
            y_encoded = le_y.fit_transform(y)

        # Encode dữ liệu đặc trưng
        X_encoded, _ = encode_data(X)

        # Khởi tạo và huấn luyện cây quyết định
        tree = ID3DecisionTree(criterion=criterion)
        tree.fit(X_encoded, y_encoded, feature_names=selected_columns)

        # Cập nhật class_names với nhãn gốc
        tree.class_names = le_y.classes_

        # Lấy thông tin cây
        tree_info = tree.export_tree()

        # Visualize cây quyết định bằng plot_tree()
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(tree.model, feature_names=selected_columns, class_names=list(map(str, tree.class_names)), ax=ax)

        # Chuyển đổi figure sang định dạng SVG
        buf = BytesIO()
        plt.savefig(buf, format='svg')
        tree_svg = base64.b64encode(buf.getvalue()).decode('utf-8')

        plt.close(fig) 
        print(tree_info) 

        # Chuẩn bị trả về kết quả
        return {
            "tree_graph": tree_svg,
            "tree_info": tree_info
        }
    except Exception as e:
        # Nếu có lỗi xảy ra, trả về thông báo lỗi
        import traceback
        print(traceback.format_exc())  # In chi tiết lỗi để debug
        return {"error": f"\u0110\u00e3 x\u1ea3y ra l\u1ed7i: {str(e)}"}, 500

if __name__ == '__main__':
    # Đảm bảo thư mục upload tồn tại
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)




# Để thêm thuật toán mới, chỉ cần tạo file Python trong 
# algorithms/ và thêm logic xử lý vào app.py và algorithm.html.
# Có thể thêm tính năng trực quan hóa kết quả (biểu đồ, bảng) vào giao diện từng thuật toán.