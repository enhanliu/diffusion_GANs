import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
import model
import diffusion_data as da
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

train_path = ' '
val_path = ' '
test_path = ' '
weights_save_path = ' '
results_save_path = ' '

model_path = ''
start_epoch = 2
paired_dataset_train = da.PairedData(train_path)
paired_dataset_val = da.TripleData(val_path)
paired_dataset_test = da.TripleData(test_path)
generator = model.UnetGenerator()
if model_path != '':
    generator.set_dict(paddle.load(model_path))
else:
    start_epoch = 0
# 超参数
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 100

# 优化器
optimizerG = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=generator.parameters(),
    beta1=0.5,
    beta2=0.999)

# 损失函数
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

# dataloader
data_loader_train = DataLoader(
    paired_dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
data_loader_val = DataLoader(
    paired_dataset_val,
    batch_size=BATCH_SIZE
)
data_loader_test = DataLoader(
    paired_dataset_test,
    batch_size=BATCH_SIZE
)

os.makedirs(results_save_path, exist_ok=True)  # 保存每个epoch的测试结果

os.makedirs(weights_save_path, exist_ok=True)  # 保存模型
best_epoch = 0
best_DSC = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_iou = 0
for epoch in range(start_epoch, EPOCHS):
    print(f'***********{epoch+1}******************')
    for data in tqdm(data_loader_train):
        real_A, real_B = data  # A为有噪声 B为噪声图（无波形）
        # print(real_A.shape, real_B.shape)
        optimizerG.clear_grad()
        # D(fake)
        fake_B = generator(real_A)
        # real_B = paddle.where(real_B >= 0,
        #                           paddle.ones_like(real_B),
        #                           -paddle.ones_like(real_B))

        g_l1_loss = mse_loss(fake_B, real_B)
        g_loss = g_l1_loss

        # train G
        g_loss.backward()
        optimizerG.step()
        # break

    print(f'Epoch [{epoch + 1}/{EPOCHS}] , Loss G: {g_loss.numpy()}')

    paddle.save(generator.state_dict(),
                os.path.join(weights_save_path,
                             train_path.split('/')[-2] + '_unet_epoch' + str(epoch + 1).zfill(3) + '.pdparams'))
    # test
    generator.eval()
    with paddle.no_grad():
        # val
        DSC = 0
        precision = 0
        recall = 0
        f1_value = 0
        iou = 0
        for data in tqdm(data_loader_val):
            real_A, real_B, real_C = data  # b为心电波形，c为噪声背景
            fake_C = generator(real_A)  # 声测的噪声背景

            fake_D = fake_C - real_A
            # Reconstruct fake_B from fake_C and real_A
            fake_B_np = real_A.numpy()
            fake_D_np = fake_D.numpy()
            base_mask = (np.abs(fake_D_np) - 1) < 0
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
            real_B = paddle.where(real_B < 0, ones, zeros)
            fake_B = paddle.where(fake_B < 0, ones, zeros)
            # real_B = paddle.where(real_B < 0, zeros, ones)
            # fake_B = paddle.where(fake_B < 0, zeros, ones)
            fake_jiao_real = paddle.where(paddle.logical_and(real_B == fake_B, real_B == 1), ones, zeros)
            fake_huo_real = paddle.where(paddle.logical_or(real_B == 1, fake_B == 1), ones, zeros)
            # print('real_B:',real_B.sum())
            # print('fake_B:',fake_B.sum())
            # print('fake_jiao_real:',fake_jiao_real.sum())
            # print('fake_huo_real:',fake_huo_real.sum())

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

        DSC = DSC / len(paired_dataset_val)
        precision = precision / len(paired_dataset_val)
        recall = recall / len(paired_dataset_val)
        iou = iou / len(paired_dataset_val)
        f1_value = 2 * precision * recall / (precision + recall)
        print(f"DSC: {DSC:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_value:.4f}")
        # test
        DSC = 0
        precision = 0
        recall = 0
        f1_value = 0
        iou = 0
        for data in tqdm(data_loader_test):
            real_A, real_B, real_C = data  # b为心电波形，c为噪声背景
            fake_C = generator(real_A)  # 声测的噪声背景

            fake_D = fake_C - real_A
            # Reconstruct fake_B from fake_C and real_A
            fake_B_np = real_A.numpy()
            fake_D_np = fake_D.numpy()
            base_mask = (np.abs(fake_D_np) - 1) < 0
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
            real_B = paddle.where(real_B < 0, ones, zeros)
            fake_B = paddle.where(fake_B < 0, ones, zeros)
            # real_B = paddle.where(real_B < 0, zeros, ones)
            # fake_B = paddle.where(fake_B < 0, zeros, ones)
            fake_jiao_real = paddle.where(paddle.logical_and(real_B == fake_B, real_B == 1), ones, zeros)
            fake_huo_real = paddle.where(paddle.logical_or(real_B == 1, fake_B == 1), ones, zeros)
            # print('real_B:',real_B.sum())
            # print('fake_B:',fake_B.sum())
            # print('fake_jiao_real:',fake_jiao_real.sum())
            # print('fake_huo_real:',fake_huo_real.sum())

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

        DSC = DSC / len(paired_dataset_test)
        precision = precision / len(paired_dataset_test)
        recall = recall / len(paired_dataset_test)
        iou = iou / len(paired_dataset_test)
        f1_value = 2 * precision * recall / (precision + recall)
        print(f"DSC: {DSC:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_value:.4f}")
        # fake_B = generator(real_A)

    if best_f1 < f1_value:
        best_epoch = epoch
        best_DSC = DSC
        best_precision = precision
        best_recall = recall
        best_f1 = f1_value
        best_iou = iou
    result = paddle.concat([real_A[:3], real_C[:3], fake_C[:3], -real_B[:3], -fake_B[:3]], 3)
    result = result.detach().numpy().transpose(0, 2, 3, 1)
    result = np.vstack(result)
    result = (result * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(
        os.path.join(results_save_path, train_path.split('/')[-2] + '_unet_epoch' + str(epoch + 1).zfill(3) + '.png'),
        result)
    generator.train()
    print(f"best_epoch: {best_epoch} *************###############################")
    print(f"best_DSC: {best_DSC:.4f}")
    print(f"best_IoU: {best_iou:.4f}")
    print(f"best_Precision: {best_precision:.4f}")
    print(f"best_Recall: {best_recall:.4f}")
    print(f"best_F1 Score: {best_f1:.4f}")