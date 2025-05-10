import cv2
import numpy as np
from scipy import ndimage
import os


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def calc_artifact_map(img, img2, crop_border, input_order='HWC', window_size=11, **kwargs):
    """Calculate quantitative indicator in Equation 7.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: artifact map between two images.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    artifact_maps = []
    for i in range(img.shape[2]):
        indicator = calc_single_artifact_map(img[..., i], img2[..., i], window_size)
        artifact_maps.append(indicator)

    artifact_maps = np.stack(artifact_maps, axis=0)
    # mean
    artifact_map = np.mean(artifact_maps, axis=0)

    return artifact_map


def calc_single_artifact_map(img, img2, window_size=11):
    """The proposed quantitative indicator in Equation 7.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: artifact map of a single channel.
    """

    constant = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[window_size // 2:-(window_size // 2),
          window_size // 2:-(window_size // 2)]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[window_size // 2:-(window_size // 2), window_size // 2:-(window_size // 2)]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[window_size // 2:-(window_size // 2),
                window_size // 2:-(window_size // 2)] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[window_size // 2:-(window_size // 2),
                window_size // 2:-(window_size // 2)] - mu2_sq

    contrast_map = (2 * (sigma1_sq + 1e-8) ** 0.5 * (sigma2_sq + 1e-8) ** 0.5 + constant) / (
            sigma1_sq + sigma2_sq + constant)

    return contrast_map


def calc_mask(mse_img_path, hr_img_path, save_final_mask_path1, save_final_mask_path2, window_size=11,
              contrast_threshold=0.7, area_threshold=4000):
    mse_img = cv2.imread(mse_img_path, cv2.IMREAD_UNCHANGED)
    gan_img = cv2.imread(hr_img_path, cv2.IMREAD_UNCHANGED)
    artifact_map = calc_artifact_map(mse_img, gan_img, crop_border=0, window_size=window_size)
    contrast_seg_mask = np.zeros(artifact_map.shape)
    contrast_seg_mask[artifact_map < contrast_threshold] = 1
    # artifact_map[artifact_map < contrast_threshold] = 1
    artifact_map = contrast_seg_mask

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(artifact_map, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=3)
    dst = ndimage.binary_fill_holes(dilation, structure=np.ones((3, 3))).astype(int)
    cv2.imwrite(save_final_mask_path1, dst * 255)

    dst = cv2.imread(save_final_mask_path1, cv2.IMREAD_GRAYSCALE)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=8)
    image_filtered = np.zeros_like(dst)
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue
        if stats[i][-1] > area_threshold:
            image_filtered[labels == i] = 255

    cv2.imwrite(save_final_mask_path2, image_filtered)


def process_folders(folder_a, folder_b, folder_c, folder_d):
    """
    处理A和B文件夹中的图片对，将结果保存到C和D文件夹

    参数:
        folder_a: 包含A图像的文件夹路径
        folder_b: 包含B图像的文件夹路径
        folder_c: 保存第一个mask结果的文件夹路径
        folder_d: 保存第二个mask结果的文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(folder_c, exist_ok=True)
    os.makedirs(folder_d, exist_ok=True)

    # 获取A文件夹中的所有图片文件
    a_images = [f for f in os.listdir(folder_a) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    b_images = [f for f in os.listdir(folder_b) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_namea, img_nameb in zip(a_images, b_images):
        # 构建对应文件的路径
        img_a_path = os.path.join(folder_a, img_namea)
        img_b_path = os.path.join(folder_b, img_nameb)

        # 检查B文件夹中是否存在对应文件
        if not os.path.exists(img_b_path):
            print(f"警告: {img_namea} 在B文件夹中不存在，跳过处理")
            continue

        # 构建输出路径
        mask_c_path = os.path.join(folder_c, img_namea)
        mask_d_path = os.path.join(folder_d, img_namea)

        # 调用cal_mask函数处理
        calc_mask(img_a_path, img_b_path, mask_c_path, mask_d_path, window_size=11, contrast_threshold=0.7,
                  area_threshold=4000)
        print(f"已处理: {img_namea},{img_nameb}")


if __name__ == '__main__':
    folder_a = "results/HAT_SRx4_loveda_mse/visualization/lovedatest"  # A文件夹路径
    folder_b = "testdata/Loveda_1024_pro/HR"  # B文件夹路径
    folder_c = "mask1"  # C输出文件夹路径
    folder_d = "mask2"  # D输出文件夹路径

    # 处理所有图片
    process_folders(folder_a, folder_b, folder_c, folder_d)
    print("所有图片处理完成!")

    window_size = 11
    contrast_threshold = 0.7
    area_threshold = 4000

# TODO: 记得对边缘进行虚化！！