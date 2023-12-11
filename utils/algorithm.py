import cv2
import numpy as np
import math

# paddlers
from paddlers.tasks import load_model
from paddlers import transforms as T


# 导向滤波
def guidedFilter(I, p, r=81, eps=1e-8):
    # 计算均值
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    mean_II = cv2.boxFilter(I * I, -1, (r, r))

    # 计算协方差
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    # 计算滤波器的权重参数
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 计算最终的滤波结果
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return mean_a * I + mean_b


# 暗通道先验的去雾算法
def darkChannelPriorDehaze(image):
    # 将图像归一化到[0,1]范围内
    if np.max(image) > 1:
        image = image / 255.0

    # 结果集
    result = np.zeros(image.shape)

    # 参数设置
    w = 0.95

    # 计算暗通道
    atmo_mask = np.min(image, 2)
    dark_channel = cv2.erode(atmo_mask, np.ones((15, 15)))
    atmo_mask = guidedFilter(atmo_mask, dark_channel)

    # 计算图像的直方图
    histogram, bins = np.histogram(atmo_mask, 2000)
    # 计算累积分布函数 (CDF)
    cdf = histogram.cumsum()
    # 从暗通道Idark中选取前0.1%个点
    for lmax in range(1999, 0, -1):
        if cdf[lmax] <= 0.999:
            break
    # 对应到有雾影像I中去求前1%最亮点的均值，作为全局大气光的估计值A
    atmo_illum = np.mean(image, 2)[atmo_mask >= histogram[lmax]].max()
    atmo_mask = np.minimum(atmo_mask * w, 0.80)

    for k in range(3):
        result[:, :, k] = (image[:, :, k] - atmo_mask) / (1 - atmo_mask / atmo_illum)

    # 限制输出范围并返回
    result = np.clip(result, 0, 1)
    return (result * 255).astype("uint8")


# butterworth滤波器
def butterworthFilter(I_shape, filter_params):
    P = I_shape[0] / 2
    Q = I_shape[1] / 2
    U, V = np.meshgrid(
        range(I_shape[0]), range(I_shape[1]), sparse=False, indexing="ij"
    )
    Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
    H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
    return 1 - H


# 同态滤波去雾算法
def homomorphicFilterDeHaze(image):
    if np.max(image) > 1:
        image = image / 255.0

    a = 0.75
    b = 1.25

    I_filtered = np.zeros_like(image)
    for i in range(3):
        I_single_channel = image[:, :, i]
        I_log = np.log1p(np.array(I_single_channel, dtype="float"))
        I_fft = np.fft.fft2(I_log)
        H = butterworthFilter(I_shape=I_fft.shape, filter_params=[30, 2])

        H = np.fft.fftshift(H)
        I_fft_filt = (a + b * H) * I_fft
        I_filt = np.fft.ifft2(I_fft_filt)
        I_single_channel_filtered = np.exp(np.real(I_filt)) - 1
        I_filtered[:, :, i] = I_single_channel_filtered
    return (I_filtered * 255).astype("uint8")


# 线性拉伸处理
# 去掉最大最小0.5%的像素值 线性拉伸至[0,1]
def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins)
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


# 根据半径计算权重参数矩阵
g_para = {}


def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h**2 + w**2)
    m /= m.sum()
    g_para[radius] = m
    return m


# 常规的ACE实现
def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh = []
    zw = []
    n = 0
    while n < radius:
        zh.append(0)
        zw.append(0)
        n += 1
    for n in range(height):
        zh.append(n)
    for n in range(width):
        zw.append(n)
    n = 0
    while n < radius:
        zh.append(height - 1)
        zw.append(width - 1)
        n += 1

    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += para[h][w] * np.clip(
                (I - Z[h : h + height, w : w + width]) * ratio, -1, 1
            )
    return res


# 单通道ACE快速增强实现
def zmIceFast(I, ratio, radius):
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) / 2), int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


# rgb三通道分别增强 ratio是对比度增强因子 radius是卷积模板半径
def zmIceColor(I, ratio=4, radius=3):
    if np.max(I) > 1:
        I = I / 255.0

    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return (res * 255).astype("uint8")


# 单波段直方图均衡化
def equalizeGray(image):
    # 计算图像的直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256], density=True)

    # 计算累积分布函数 (CDF)
    cdf = hist.cumsum()

    # 创建一个映射函数，将原始像素值映射到均衡化后的像素值
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = np.round(cdf_m * 255)
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")

    # 使用映射函数来均衡化图像
    equalized_image = cdf[image]

    return equalized_image


# 直方图均衡化去雾算法
def equalizeHistDehaze(image):
    # 分离图像的三个颜色通道
    b, g, r = cv2.split(image)

    # 对每个颜色通道进行直方图均衡化
    r_eq = equalizeGray(r)
    g_eq = equalizeGray(g)
    b_eq = equalizeGray(b)

    # 将均衡化后的颜色通道合并回一个图像
    equalized_image = cv2.merge([b_eq, g_eq, r_eq])

    return equalized_image


# Retinex算法单通道处理
def retinex_channel_processing(channel, sigma):
    channel_log = np.log1p(channel)
    channel_fft2 = np.fft.fft2(channel)

    # 创建高斯滤波器
    N, M = channel.shape
    F = np.fft.fft2(cv2.getGaussianKernel(N, sigma) * cv2.getGaussianKernel(M, sigma).T)

    # 应用滤波器
    channel_fft2_filtered = channel_fft2 * F

    # 傅里叶逆变换得到处理后的通道
    channel_filtered = np.fft.ifft2(channel_fft2_filtered).real

    # 计算对数域的差异
    channel_retinex = channel_log - np.log1p(channel_filtered)

    # 指数得出最终结果
    channel_result = np.exp(channel_retinex)

    # 标准化 [0, 1]
    channel_result = (channel_result - np.min(channel_result)) / (
        np.max(channel_result) - np.min(channel_result)
    )

    # 应用自适应直方图均衡
    channel_result = (
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            (channel_result * 255).astype(np.uint8)
        )
        / 255.0
    )

    return channel_result


# Retinex算法去雾
def retinexDehaze(image):
    if np.max(image) > 1:
        image = image / 255.0

    # 分离图像的三个颜色通道
    b, g, r = cv2.split(image)

    # 对每个颜色通道进行Retinex处理
    r_retinex = retinex_channel_processing(r, 10)
    g_retinex = retinex_channel_processing(g, 10)
    b_retinex = retinex_channel_processing(b, 10)

    # 将处理后的颜色通道合并回一个图像
    retinex_image = cv2.merge([b_retinex, g_retinex, r_retinex])

    return (retinex_image * 255).astype("uint8")


# NAFNET预测
model = load_model("./data/best_model")
eval_transforms = [
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
]


def nafnetPredict(img):
    result = model.predict(img, T.Compose(eval_transforms))["res_map"]
    return result
    return img