from paddle.io import Dataset, DataLoader
import os
import cv2
from PIL import Image
import numpy as np
from paddle.vision import transforms
import paddle


def override_black_with_white(base_path, mask_path, output_path='masked_result.png'):
    # 加载两张图像并转换为灰度
    base_img = Image.open(base_path).convert('L')
    mask_img = Image.open(mask_path).convert('L')

    # 确保尺寸一致（可根据需要裁剪或缩放，这里取最小公共区域裁剪）
    min_width = min(base_img.width, mask_img.width)
    min_height = min(base_img.height, mask_img.height)

    base_array = np.array(base_img.crop((0, 0, min_width, min_height)))
    mask_array = np.array(mask_img.crop((0, 0, min_width, min_height)))

    # 将 mask 中为黑色（值为 0）的位置，在 base 中设为白色（255）
    base_array[mask_array == 0] = 255


class TripleData(Dataset):
    def __init__(self, phase):
        super(TripleData, self).__init__()
        self.img_path_list = self.load_A2B_data(phase)  # 获取数据列表
        self.num_samples = len(self.img_path_list)  # 数据量
        self.transform = transforms.Compose([
            transforms.Resize((1024, 2048)),  # 替换new_height和new_width为你需要的大小
            transforms.ToTensor()  # 如果需要转换为tensor的话
        ])

    def __getitem__(self, idx):
        # 获取带噪声心电图
        img_A2B = cv2.imread(self.img_path_list[idx])  # 读取数据
        img_A2B = cv2.resize(img_A2B, (2048, 1024))
        # 获取纯净心电图图像
        path_b = os.path.join('work/ecg_pro', os.path.split(self.img_path_list[idx])[1].split('_')[0] + '.png')
        # print(path_b)
        img_B = cv2.imread(path_b)
        img_B = cv2.resize(img_B, (2048, 1024))

        # 构建无ECG噪声
        img_C = img_A2B.copy()
        mask = np.all(img_B < 255, axis=2)
        img_C[mask] = 255

        # 噪声心电图转为单通道：
        img_A2B = cv2.cvtColor(img_A2B, cv2.COLOR_BGR2GRAY)
        img_A2B = np.expand_dims(img_A2B, axis=2)
        img_A2B = img_A2B.transpose(2, 0, 1)  # HWC -> CHW
        img_A2B = img_A2B.astype('float32') / 127.5 - 1.  # 归一化
        img_A = img_A2B

        # 纯净心电图转为单通道
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        img_B = np.expand_dims(img_B, axis=2)
        img_B = img_B.transpose(2, 0, 1)
        img_B = img_B.astype('float32') / 127.5 - 1.

        img_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)
        img_C = np.expand_dims(img_C, axis=2)
        img_C = img_C.transpose(2, 0, 1)
        img_C = img_C.astype('float32') / 127.5 - 1.
        # 此时获取了带噪声图像A 和纯净心电图B
        # 现在通过计算A和B之间的差异来获取A中的背景噪声作为新的标签图C
        # 标签图C中包含噪点的位置信息，噪点位置为1，其他位置为0
        # 也就是说A和B异或 结果为图C

        # resize
        # img_A = self.transform(img_A)
        # img_B = self.transform(img_B)
        return img_A, img_B, img_C  # A为有噪声，B为无噪声，C是噪声位置图
        # return img_A[:, :255, :255], img_B[:, :255, :255]

    def __len__(self):
        return self.num_samples

    @staticmethod
    def load_A2B_data(phase):
        data_path = phase
        return [os.path.join(data_path, x) for x in os.listdir(data_path)]


class PairedData(Dataset):
    def __init__(self, phase, noise_type=''):
        super(PairedData, self).__init__()
        self.img_path_list = self.load_A2B_data(phase)  # 获取数据列表
        self.num_samples = len(self.img_path_list)  # 数据量
        self.transform = transforms.Compose([
            transforms.Resize((1024, 2048)),  # 替换new_height和new_width为你需要的大小
            transforms.ToTensor()  # 如果需要转换为tensor的话
        ])
        self.noise_type = noise_type

    def __getitem__(self, idx):
        # 获取带噪声心电图
        img_A2B = cv2.imread(self.img_path_list[idx])  # 读取数据
        img_A2B = cv2.resize(img_A2B, (2048, 1024))
        # 获取纯净心电图图像
        path_b = os.path.join('work/ecg_pro', os.path.split(self.img_path_list[idx])[1].split('_')[0] + '.png')
        # print(path_b)
        img_B = cv2.imread(path_b)
        img_B = cv2.resize(img_B, (2048, 1024))
        # 构建无ECG噪声
        img_C = img_A2B.copy()
        mask = np.all(img_B < 255, axis=2)
        img_C[mask] = 255

        # 噪声心电图转为单通道：
        if len(img_A2B.shape) == 3 and img_A2B.shape[2] == 3:
            img_A2B = cv2.cvtColor(img_A2B, cv2.COLOR_BGR2GRAY)
        img_A2B = np.expand_dims(img_A2B, axis=2)
        img_A2B = img_A2B.transpose(2, 0, 1)  # HWC -> CHW
        img_A2B = img_A2B.astype('float32') / 127.5 - 1.  # 归一化
        # img_A = img_A2B[..., :256]                        # 真人照
        img_A = img_A2B

        # 纯净心电图转为单通道
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        img_B = np.expand_dims(img_B, axis=2)
        img_B = img_B.transpose(2, 0, 1)
        img_B = img_B.astype('float32') / 127.5 - 1.

        if len(img_C.shape) == 3:
            img_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)
        img_C = np.expand_dims(img_C, axis=2)
        img_C = img_C.transpose(2, 0, 1)
        img_C = img_C.astype('float32') / 127.5 - 1.
        # 此时获取了带噪声图像A 和纯净心电图B
        # 现在通过计算A和B之间的差异来获取A中的背景噪声作为新的标签图C
        # 标签图C中包含噪点的位置信息，噪点位置为1，其他位置为0
        # 也就是说A和B异或 结果为图C

        # resize
        # img_A = self.transform(img_A)
        # img_B = self.transform(img_B)
        return img_A, img_C  # A为有噪声，B为无噪声，C是噪声位置图
        # return img_A[:, :255, :255], img_B[:, :255, :255]

    def __len__(self):
        return self.num_samples

    @staticmethod
    def load_A2B_data(phase):
        data_path = phase
        return [os.path.join(data_path, x) for x in os.listdir(data_path)]