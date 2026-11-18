'''=================================================
@Author : Liang Wu
@Email  : liangwu1998@163.com
@Date   : 2025/4/24 11:29
=================================================='''
import scipy.io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from scipy.ndimage import median_filter
# from scipy.signal import medfilt2d
# from skimage.filters import threshold_otsu
import numpy as np


def import_mat_data(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    return next(v for k, v in mat_data.items() if not k.startswith('__'))


def quality_index(ori_data, obs_data):
    PSNR = peak_signal_noise_ratio(ori_data, obs_data, data_range=1)
    SSIM = structural_similarity(ori_data, obs_data, use_sample_covariance=False, channel_axis=2, data_range=1.0,
                                 gaussian_weights=True)
    RE = np.sqrt(np.sum(np.square(ori_data - obs_data)) / np.sum(np.square(ori_data)))
    return PSNR, SSIM, RE


# def binary_data(input_data):
#     [n1, n2, n3] = input_data.shape
#     binary_result = np.zeros_like(input_data, dtype=np.float32)
#     for i in range(n3):
#         A = input_data[:, :, i]
#         A_uint8 = np.round(A * 255).astype(np.uint8)
#         level = threshold_otsu(A_uint8)
#         binary_result[:, :, i] = (A_uint8 > level).astype(np.float32)
#     return  binary_result

def binary_data(input_data):
    beta = np.std(input_data, axis=(0, 1)) ** 2
    O = (input_data ** 2 > beta[np.newaxis, np.newaxis, :]).astype(np.float64)
    binary_result = median_filter(O, size=(3, 3, 1), mode='constant', cval=0)
    return binary_result


# def binary_data(input_data):
#     M, N, P = input_data.shape
#     # 计算每个切片的阈值
#     beta = np.std(input_data, axis=(0, 1)) ** 2
#
#     # 预分配输出
#     binary_result = np.zeros((M, N, P), dtype=np.float64)
#
#     for k in range(P):
#         # 硬阈值
#         O = (input_data[:, :, k] ** 2 > beta[k]).astype(np.float64)
#         # 中值滤波，边界零填充
#         binary_result[:, :, k] = medfilt2d(O, kernel_size=(3, 3))
#     return binary_result


def detection_index(gt_data, input_data):
    correct = np.sum(gt_data * input_data, axis=(0, 1))  # shape: (n3,)
    est_sum = np.sum(input_data, axis=(0, 1))  # 每个切片预测为正的个数
    truth_sum = np.sum(gt_data, axis=(0, 1))  # 每个切片真实为正的个数

    # 2) Precision 和 Recall，除零时置为 0
    p = np.where(est_sum != 0, correct / est_sum, 0.0)
    r = np.where(truth_sum != 0, correct / truth_sum, 0.0)

    # 3) F-measure，分母为零时置为 0
    sum_pr = p + r
    f = np.where(sum_pr != 0, 2 * p * r / sum_pr, 0.0)

    # 4) 取最大 F 及其索引，和平均值
    f_max_index = int(np.argmax(f))
    f_max_value = float(f[f_max_index])
    f_mean = float(np.mean(f))

    return f_max_value, f_max_index, f_mean


# def detection_index(gt_data, input_data):
#     [n1, n2, n3] = input_data.shape
#     # recall_list = []
#     # precision_list = []
#     fmeasure_list = []
#     for i in range(n3):
#         truth = gt_data[:, :, i]
#         est = input_data[:, :, i]
#         correct = np.sum((truth + est) == 2)
#         p = correct / np.sum(est) if np.sum(est) != 0 else 0  # 精确率
#         r = correct / np.sum(truth) if np.sum(truth) != 0 else 0  # 召回率
#         f = (2*r*p)/(r+p)
#         # recall_list.append(r)
#         # precision_list.append(p)
#         fmeasure_list.append(f)
#     f_max_value, f_max_index = max((val, idx) for idx, val in enumerate(fmeasure_list))
#     return f_max_value, f_max_index, np.mean(fmeasure_list), fmeasure_list


# A = import_mat_data('Curtain_mask.mat')
# B = import_mat_data('Curtain_CB50.mat')
# C = import_mat_data('Curtain.mat')
#
# D = np.abs(B - C)
# D_binary = binary_data(D)
#
# a, b, c, ff = detection_index(A, D_binary)
#
# f = import_mat_data('f.mat')
# f_mean = np.mean(f)
#
# w = np.max(np.abs(f - ff))
