import pywt
import torch
import torch.nn.functional as F
import numpy as np




def apply_wavelet_transform(tensor):
    # 使用 Haar 小波进一步细化特征
    wavelet = 'haar'
    batch_size, height, width = tensor.shape
    transformed_low_features = []
    wavelet1 = pywt.Wavelet(wavelet)
    for i in range(batch_size):
        img = tensor[i].cpu().numpy()
        coeffs1 = pywt.dwt2(img, wavelet1)
        cA1, (cH1, cV1, cD1) = coeffs1
        High = np.stack([cH1, cV1, cD1], axis=-1)
        High1 = np.mean(High, axis=-1)
        # High1 = np.where(High1 > 0.01, High1, 1e-6)
        low_frequency = cA1 * 0.9 + High1 * 0.1
        transformed_low_features.append(low_frequency)
    # 将特征转换回 PyTorch 张量
    transformed_features_low = torch.tensor(np.array(transformed_low_features), dtype=tensor.dtype, device=tensor.device)
    # 插值调整回原始维度 # cA的尺寸可能会小于原始尺寸，这里做一个简单的插值处理
    transformed_features_low_1 = F.interpolate(transformed_features_low.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)
    transformed_features_low_1 = transformed_features_low_1.squeeze(1)
    # 对融合特征和细化特征进行最终加权融合
    output_features = transformed_features_low_1
    return output_features