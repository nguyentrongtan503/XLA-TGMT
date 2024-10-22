B1: Input :Ảnh X-quang đầu vào là ảnh đơn sắc (grayscale) với các giá trị cường độ nằm trong khoảng từ 0 (màu đen) -> 255 (màu trắng)

B2: Tạo ảnh âm tính:ảnh mà các giá trị pixel của ảnh xám được đảo ngược (các vùng sáng trong ảnh gốc sẽ trở nên tối và ngược lại...)
-)Đảo ngược giá trị pixel với phép tính 255 - image.
    Cụ thể: Negative(x,y)=255−Original(x,y)
    +)Original(x, y): giá trị pixel của ảnh gốc tại vị trí (x,y). Màu giá trị này nằm trong khoảng từ 0 -> 255 
    +)Negative(x, y): giá trị pixel của ảnh âm tính tại vị trí(x,y).
    +)255: là giá trị tối đa của pixel trong ảnh 8-bit

B3: Tăng cường độ tương phản: Sử dụng cv2.convertScaleAbs với tham số alpha để điều chỉnh độ tương phản.
    -)alpha:là hệ số điều chỉnh độ sáng hoặc độ tương phản của bức ảnh
        +) alpha > 1, hình ảnh sẽ trở nên sáng hơn => tăng độ tương phản.
        +) 0 < alpha < 1, hình ảnh sẽ tối hơn => giảm độ tương phản.
        +) example: alpha = 1.5 có nghĩa là độ tương phản của bức ảnh sẽ được tăng lên 50%.

    -)beta:là hằng số điều chỉnh độ sáng tổng thể của bức ảnh
        +) beta > 0 => bức ảnh sẽ sáng hơn.
        +) beta < 0 => bức ảnh sẽ tối hơn.

    -)Phép toán Hàm cv2.convertScaleAbs trên mỗi pixel: 
           new_value= clip(α x old_value + β)
        +)clip: Giới hạn giá trị pixel trong khoảng [0, 255].(nếu giá trị mới lớn hơn 255 nó sẽ được đặt thành 255; nếu nhỏ hơn 0 nó sẽ được đặt thành 0 )

B4: Biến đổi logarit: Phép biến đổi log làm sáng các vùng tối và giữ nguyên các vùng sáng, tạo ra sự tương phản tốt hơn.
    ' c = 255 / np.log(1 + np.max(image))  
      log_image = c * (np.log(image + 1))
      log_image = np.array(log_image, dtype=np.uint8) '

      +)c: Hằng số này được tính để chuẩn hóa kết quả của phép biến đổi logarit trong khoảng giá trị pixel [0, 255]
      +)log_image: Nhân với hằng số c để đảm bảo rằng giá trị pixel nằm trong khoảng từ 0 đến 255. Kiểu dữ liệu uint8, giới hạn giá trị pixel trong khoảng [0, 255] nếu vượt quá khoảng

B5: Cân bằng histogram: Sử dụng cv2.equalizeHist để cải thiện độ tương phản bằng cách phân phối lại các mức xám của ảnh.
    Biểu diễn tần số xuất hiện của mỗi mức độ sáng (giá trị pixel) trong ảnh. Ảnh xám có giá trị pixel từ 0 đến 255, nên histogram sẽ là một biểu đồ có 256 cột, mỗi cột biểu diễn số lượng pixel có một giá trị độ sáng nhất định ,"trải đều" giá trị của các pixel trong ảnh