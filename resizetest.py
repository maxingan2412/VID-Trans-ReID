from PIL import Image
import matplotlib.pyplot as plt

def resize_and_show(image_path, new_width, new_height):
    # 打开图片
    img = Image.open(image_path)

    # 调整图片大小
    resized_img = img.resize((new_width, new_height))

    # 显示原始图片和调整后的图片
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Resized Image")
    plt.imshow(resized_img)

    plt.show()

# 测试程序
# 替换下面的 'path/to/your/image.jpg' 为您的图片路径
# 设置您想要的新宽度和高度
resize_and_show('0001C1T0001F001.jpg', 224, 224)
