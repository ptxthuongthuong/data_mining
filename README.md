Hướng dẫn Cài đặt Dự án Flask Python

1. Cài đặt Python
- Đảm bảo Python 3.8 trở lên đã được cài đặt.  
  Kiểm tra phiên bản Python:  
  python --version

2. Tạo môi trường ảo
- Tạo môi trường ảo để quản lý các thư viện:  
  python -m venv venv
- Kích hoạt môi trường ảo:
  - Windows:  
    venv\\Scripts\\activate
  - macOS/Linux:  
    source venv/bin/activate

3. Cài đặt thư viện
- Cài đặt các thư viện cần thiết từ requirements.txt:  
  pip install -r requirements.txt

4. Cấu hình thư mục
- Đảm bảo các thư mục cần thiết đã tồn tại:
  - uploads/: Lưu các file người dùng tải lên.
  - static/ và templates/: Lưu CSS, JS và HTML.

5. Chạy ứng dụng
- Chạy ứng dụng Flask:  
  python app.py
- Ứng dụng sẽ chạy trên http://127.0.0.1:5000. Mở trình duyệt và truy cập URL này.

6. Kiểm tra hệ thống
- Đảm bảo ứng dụng hoạt động và các tính năng như upload file, chạy thuật toán đều chạy đúng.

7. Ghi chú
- Nếu gặp lỗi, kiểm tra lại cấu hình môi trường ảo và đảm bảo Python cùng các thư viện tương thích.
