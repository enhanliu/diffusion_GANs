import paddle
from paddle.io import DataLoader
import model
import diffusion_data as da
import os
import cv2
import numpy as np
from tqdm import tqdm

path_list = os.listdir(' ')
MODEL_PATHS = [os.path.join(' ', ll) for ll in path_list]

# Test dataset path
TEST_PATH = ' '

# Where to save visual results
RESULTS_SAVE_PATH = ' '
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

# Batch size for testing
BATCH_SIZE = 8

# ----- Prepare Dataset & Dataloader -----
paired_dataset_test = da.TripleData(TEST_PATH)
data_loader_test = DataLoader(
    paired_dataset_test,
    batch_size=BATCH_SIZE
)

# ----- Model Initialization -----
generator = model.UnetGenerator()

# ----- Evaluation Loop -----
results_save_path = ' '
os.makedirs(results_save_path, exist_ok=True)
for model_path in MODEL_PATHS:
    print(f"Testing model: {model_path}")
    # Load weights
    generator.set_dict(paddle.load(model_path))
    generator.eval()

    # Metrics accumulators
    DSC = 0
    precision = 0
    recall = 0
    f1_value = 0
    iou = 0
    num_samples = len(paired_dataset_test)
    with paddle.no_grad():
        for data in tqdm(data_loader_test, desc="Testing batches"):  # real_A, real_B, real_C
            real_A, real_B, real_C = data  # b为心电波形，c为噪声背景
            fake_C = generator(real_A)  # 声测的噪声背景

            fake_D = fake_C - real_A
            # Reconstruct fake_B from fake_C and real_A
            fake_B_np = real_A.numpy()
            fake_D_np = fake_D.numpy()
            base_mask = (np.abs(fake_D_np) - 1) < 0 #) | (fake_C.numpy() < 0.9)  # 此为阈值，可以适当增大
            # refined_mask = postprocess_mask(fake_D_np)
            # mask = refined_mask
            # mask = base_mask | refined_mask
            # mask = base_mask & refined_mask
            mask = base_mask
            fake_B_np[mask] = 1
            fake_B = paddle.to_tensor(fake_B_np)

            # Binarize both real_B and fake_B
            ones = paddle.ones_like(fake_B)
            zeros = paddle.zeros_like(fake_B)
            real_B = paddle.where(real_B < 0., ones, zeros)
            fake_B = paddle.where(fake_B < 0., ones, zeros)
            # real_B = paddle.where(real_B < 0, zeros, ones)
            # fake_B = paddle.where(fake_B < 0, zeros, ones)
            fake_jiao_real = paddle.where(paddle.logical_and(real_B == fake_B, real_B == 1), ones, zeros)
            fake_huo_real = paddle.where(paddle.logical_or(real_B == 1, fake_B == 1), ones, zeros)

            # 计算DSC
            temp = 2 * paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (
                    paddle.sum(real_B, axis=(1, 2, 3)) + paddle.sum(fake_B, axis=(1, 2, 3)) + 0.000000001)
            # print(temp.sum())
            DSC += paddle.sum(temp).item()
            # 计算precision
            precision += paddle.sum(
                paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (paddle.sum(fake_B, axis=(1, 2, 3)) + 0.000000001)).item()
            # 计算recall
            recall += paddle.sum(
                paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (paddle.sum(real_B, axis=(1, 2, 3)) + 0.000000001)).item()
            # 计算IOU
            iou += paddle.sum(
                paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (
                        paddle.sum(fake_huo_real, axis=(1, 2, 3)) + 0.000000001)).item()

    # Final averaging
    DSC = DSC / len(paired_dataset_test)
    precision = precision / len(paired_dataset_test)
    recall = recall / len(paired_dataset_test)
    iou = iou / len(paired_dataset_test)
    f1_value = 2 * precision * recall / (precision + recall)

    # Print results
    print(f"Results for {os.path.basename(model_path)}:")
    print(f"  DSC: {DSC:.4f}")
    print(f"  IoU: {iou:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_value:.4f}\n")
    result = paddle.concat([real_A[:3], real_C[:3], fake_C[:3], -real_B[:3], -fake_B[:3]], 3)
    result = result.detach().numpy().transpose(0, 2, 3, 1)
    result = np.vstack(result)
    result = (result * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(
        os.path.join(results_save_path, model_path.split('/')[-1].split('.')[0] + '.png'),
        result)



