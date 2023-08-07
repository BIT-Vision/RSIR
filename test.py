import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
import structure
import netloss


def load_spike_numpy(path: str) -> (np.ndarray, np.ndarray):
    '''
    Load a spike sequence with it's tag from prepacked `.npz` file.\n
    The sequence is of shape (`length`, `height`, `width`) and tag of
    shape (`height`, `width`).
    '''
    data = np.load(path)
    seq, tag, length = data['seq'], data['tag'], int(data['length'])
    seq = np.array([(seq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return seq, tag, length


def RawToSpike(video_seq, h, w):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0, h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))

    return SpikeMatrix


def cal_para():

    lux1 = 0
    lux2 = 153
    lux3 = 335

    # load and process light data
    light_data = open("./uniformlight/light.dat", 'rb+')
    light_seq = light_data.read()
    light_seq = np.fromstring(light_seq, 'B')
    light_spike = RawToSpike(light_seq, 250, 416)[:, :, :400]  # c*h*w
    # print(light_spike.shape)
    light_img = light_spike[0:20000].mean(axis=0).astype(np.float32)
    # cv2.imwrite("./uniformlight/light.png", light_img * 255)
    D_light = 1 / light_img - 1

    # load meduim  data
    meduim_data = open("./uniformlight/medium.dat", 'rb+')
    meduim_seq = meduim_data.read()
    meduim_seq = np.fromstring(meduim_seq, 'B')
    meduim_spike = RawToSpike(meduim_seq, 250, 416)[:, :, :400]  # c*h*w
    meduim_img = meduim_spike[0:20000].mean(axis=0).astype(np.float32)
    # cv2.imwrite("./uniformlight/medium.png", meduim_img * 255)
    D_meduim = 1 / meduim_img - 1

    # load dark  data
    dark_data = open("./uniformlight/dark.dat", 'rb+')
    dark_seq = dark_data.read()
    dark_seq = np.fromstring(dark_seq, 'B')
    dark_spike = RawToSpike(dark_seq, 250, 416)[:, :, :400]  # c*h*w
    dark_img = dark_spike[0:20000].mean(axis=0).astype(np.float32)
    # cv2.imwrite("./uniformlight/dark.png", dark_img * 255)
    D_dark = 1 / (dark_img + 1e-5) - 1

    th = D_meduim / D_dark
    fo1 = (lux2 * th - lux1) / (1 - th)

    base = D_light[0, 0]
    fpn_test = base * (lux3 + fo1[0, 0]) / (D_light * (lux3 + fo1))

    q = D_light * (lux3 + fo1)

    return q, fo1, fpn_test


device = cfg.device
model = structure.MainDenoise()
model = model.to(device)

checkpoint = torch.load(cfg.test_checkpoint)
model.load_state_dict(checkpoint['model'])
best_PSNR = checkpoint['best_psnr']
print("the best PSNR is :{}".format(best_PSNR))

Q, Nd, Nl = cal_para()
q = torch.from_numpy(Q).to(device)
nd = torch.from_numpy(Nd).to(device)
nl = torch.from_numpy(Nl).to(device)

model.eval()
test_scene = ["tuk-tuk", "train", "upside-down", "varanus-cage", "walking"]
light_scale = ["0256", "0032"]
total_psnr = 0
total_ssim = 0
cnt = 0
with torch.no_grad():
    for tag in test_scene:
        for ls in light_scale:
            h, w = 250, 400
            seq, label, length = load_spike_numpy(cfg.simulated_dir+"spike-video{}-00000-light{}.npz".format(tag, ls))
            if ls == "0256":
                wins = 16
            elif ls == "0032":
                wins = 32
            for i in range(cfg.frame_num):
                noisy_img = np.ones([1, 1, 250, w], dtype=np.float32)
                noisy_img[0, 0, :, :] = seq[wins*i:wins*(i+1)].mean(axis=0).astype(np.float32)**(1/2.2)
                fgt = np.ones([1, 1, 250, w], dtype=np.float32)
                fgt[0, 0, :, :] = (label / 255.0)**(1/2.2)
                cv2.imwrite("./result/gt_{}.png".format(tag), fgt[0, 0] * 255.0)
                cv2.imwrite("./result/noisy_{}.png".format(tag), noisy_img[0, 0] * 255.0)
                noisy_img = torch.from_numpy(noisy_img).to(device)
                fgt = torch.from_numpy(fgt).to(device)
                if i == 0:
                    input = noisy_img
                else:
                    input = torch.cat([noisy_img, ft0_fusion_data], dim=1)

                fpn_denoise, img_true, fusion_out, denoise_out, refine_out, ft_denoise_out_d0, fgt_d0 = model(input, fgt, q, nd, nl)
                fgt = F.pad(fgt, [0, 0, 3, 3], mode="reflect")
                ft0_fusion_data = fusion_out[:, :, 3:253, :]
                if i == cfg.frame_num-1:
                    psnr = netloss.PSNR().to(device)
                    ssim = netloss.SSIM().to(device)
                    cnt += 1
                    PSNR = psnr(refine_out, fgt).item()
                    SSIM = ssim(refine_out, fgt).item()
                    total_psnr += PSNR
                    total_ssim += SSIM
                    print("%10s: PSNR:%.2f SSIM:%.4f" % (tag, PSNR, SSIM))

    print("Total PSNR:")
    print(total_psnr / cnt)
    print("Total SSIM:")
    print(total_ssim / cnt)
