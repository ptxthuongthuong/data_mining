from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
from algorithms.naive_bayes import naive_bayes_classifier
from algorithms.utils import update_dataset_info, encode_data
from algorithms.kmeans import preprocess_data, kmeans_clustering
from algorithms.desision_tree import ID3DecisionTree, plot_custom_decision_tree
from algorithms.apriori import apriori_algorithm
from algorithms.reduct import find_decision_class, find_equivalence_classes,lower_approximation, upper_approximation, boundary_region, outside_region, find_reducts, calculate_dependency_degree


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
        "apriori":"apriori.html",
        "reduct" : "reduct.html"
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

        if not uploaded_file or not dataset_info:
            return {"error": "Vui lòng tải lên dataset trước khi chạy thuật toán"}, 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        df = pd.read_excel(filepath)

        criterion = request.form.get('criterion', 'entropy')  # 'entropy' hoặc 'gini'
        target_column = request.form.get('target_column')
        selected_columns = request.form.getlist('selected_columns')

        if not target_column or not selected_columns:
            return {"error": "Vui lòng chọn cột mục tiêu và các cột dữ liệu"}, 400

        # Chỉ lấy cột dạng categorical
        X = df[selected_columns]
        y = df[target_column]

        if not X.select_dtypes(include='object').shape[1] == len(selected_columns):
            return {"error": "Chỉ hỗ trợ dữ liệu dạng categorical (chuỗi hoặc phân loại)"}, 400

        # Khởi tạo và huấn luyện mô hình Decision Tree
        tree = ID3DecisionTree(criterion=criterion)
        tree.fit(X, y, feature_names=selected_columns)

        # Xuất thông tin cây quyết định
        tree_info = tree.export_tree()

        # Lấy chi tiết thông số của từng node
        node_details_df = tree.export_node_details()
        
        # Thêm cột cho việc hiển thị trên giao diện
        node_details_df['Node Info'] = node_details_df.apply(
            lambda row: f"Parent: {row['parent_feature'] or 'Root'}, " + 
                        f"Value: {row['node_value']}, " + 
                        f"{'Leaf' if row['is_leaf'] else 'Internal Node'}", 
            axis=1
        )
        
        # Định dạng các cột số để dễ đọc
        numeric_columns = ['information_gain', 'entropy', 'gini']
        for col in numeric_columns:
            if col in node_details_df.columns:
                node_details_df[col] = node_details_df[col].apply(lambda x: f'{x:.4f}' if pd.notnull(x) else 'N/A')
        
        # Chuyển DataFrame sang HTML để hiển thị
        node_details_html = node_details_df.to_html(
            classes="table table-bordered table-striped", 
            index=False, 
            columns=['Node Info'] + numeric_columns + ['class_distribution']
        )

        # Visualization (SVG) cho cây quyết định
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        plt = plot_custom_decision_tree(
            tree.node,
            feature_names=selected_columns,
            class_names=list(map(str, tree.class_names))
        )

        buf = BytesIO()
        plt.savefig(buf, format='svg')
        tree_svg = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return {
            "tree_graph": tree_svg,
            "tree_info": tree_info,
            "node_details": node_details_html
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"Đã xảy ra lỗi: {str(e)}"}, 500

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


@app.route('/run_reduct', methods=['POST'])
def run_reduct():
    global dataset_info, uploaded_file

    # Get form data
    object_column = request.form.get('object_column')
    decision_column = request.form.get('decision_column')
    target_value = request.form.get('target_value')
    condition_attributes = request.form.getlist('condition_attributes')

    if not object_column or not decision_column or not target_value or not condition_attributes:
        return jsonify({"error": "Vui lòng chọn giá trị và thuộc tính."})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    df = pd.read_excel(filepath)
    
    # Find X set based on selected decision attribute
    X = find_decision_class(df, decision_column, target_value)
    
    # Find equivalence classes based on condition attributes
    IND = find_equivalence_classes(df, condition_attributes)
    
    # Calculate approximations
    upper = upper_approximation(X, IND, decision_column)
    lower = lower_approximation(X, IND, decision_column)
    boundary = boundary_region(X, upper, lower)
    outer = outside_region(X, IND, upper, lower)

    # Extract just the object IDs
    result_X = X[object_column].tolist()
    result_IND = [df[object_column].tolist() for df in IND]
    result_lower = [df[object_column].tolist() for df in lower]
    result_upper = [df[object_column].tolist() for df in upper]
    result_boundary = [df[object_column].tolist() for df in boundary]
    result_outer = [df[object_column].tolist() for df in outer]

    # Calculate dependency degree (assuming you have this function)
    # Tìm reduct và độ phụ thuộc
    reduct = find_reducts(df, condition_attributes, decision_column, X)
    dependency = calculate_dependency_degree(X,lower)
    
    # Trả kết quả dưới dạng JSON
    return jsonify({
        "success": True,
        "object_column": object_column,  # Thêm dòng này

        "X": result_X,
        "IND": result_IND,
        "upper": result_upper,
        "lower": result_lower,
        "vùng biên": result_boundary,
        "vùng ngoài": result_outer,
        "reduct": reduct,
        "dependency_degree": dependency
    })


@app.route('/get_unique_values', methods=['GET'])
def get_unique_values():
    column = request.args.get('column')
    if not column:
        return jsonify({'error': 'No column specified'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    try:
        data = pd.read_excel(filepath)
        if column not in data.columns:
            return jsonify({'error': 'Invalid column specified'}), 400

        unique_values = data[column].unique().tolist()
        return jsonify({'unique_values': unique_values})
    except Exception as e:
        print(f"Error processing column: {e}")
        return jsonify({'error': 'Failed to process the file'}), 500

if __name__ == '__main__':
    # Đảm bảo thư mục upload tồn tại
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)




# Để thêm thuật toán mới, chỉ cần tạo file Python trong 
# algorithms/ và thêm logic xử lý vào app.py và algorithm.html.
# Có thể thêm tính năng trực quan hóa kết quả (biểu đồ, bảng) vào giao diện từng thuật toán.
