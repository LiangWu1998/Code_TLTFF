'''=================================================
@Author : Liang Wu
@Email  : liangwu1998@163.com
@Date   : 2025/4/1 15:13
=================================================='''
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import itertools
import math
from tqdm import tqdm
from core_function.evaluation_function import *
from core_function.utils import *

dtype = torch.cuda.FloatTensor
os.makedirs("./results", exist_ok=True)
###############################

# Here are the hyperparameters.

w_decay_s = [2]  
omega_s = [2]  
shrink_s = [2]  
expand_s = [8]  
lr_real_s = [1e-4]  
max_iter = 6000

data_path = './data/color_image/'

ori_name_list = ['ori_Boats']

obs_name_list = ['sr5_Boats']

mark_name_list = ['mark5_Boats']

###########################

for ori_name, obs_name, mark_name in zip(ori_name_list, obs_name_list, mark_name_list):

    gt = torch.from_numpy(import_mat_data(data_path + ori_name + '.mat')).type(dtype).cuda()
    X = torch.from_numpy(import_mat_data(data_path + obs_name + '.mat')).type(dtype).cuda()
    mask = torch.from_numpy(import_mat_data(data_path + mark_name + '.mat')).type(dtype).cuda()

    gt_min = gt.min().item()
    gt_max = gt.max().item()
    [n_1, n_2, n_3] = X.shape
    n_1_input = torch.from_numpy(np.array(range(1, n_1 + 1))).reshape(n_1, 1).type(dtype)
    n_2_input = torch.from_numpy(np.array(range(1, n_2 + 1))).reshape(n_2, 1).type(dtype)
    n_3_input = torch.from_numpy(np.array(range(1, n_3 + 1))).reshape(n_3, 1).type(dtype)

    #############################
    init_param_combinations = list(itertools.product(w_decay_s, omega_s, shrink_s, expand_s, lr_real_s))
    combinations_numbers = len(init_param_combinations)
    for idx, init_param in enumerate(init_param_combinations, start=1):
        w_decay, omega, shrink, expand, lr_real = init_param
        model = Network(n_1, n_2, n_3, shrink, expand, omega).type(dtype)
        params = []
        params += [x for x in model.parameters()]
        s = sum([np.prod(list(p.size())) for p in params])
        print('Number of params: %d' % s)
        optimizier = optim.Adam(params, lr=lr_real, weight_decay=w_decay)

        pbar = tqdm(range(max_iter), desc=f"Combo {idx}/{combinations_numbers}", ncols=100)
        psnr_list = []
        ssim_list = []
        re_list = []
        for iter in pbar:
            X_Out = model(n_1_input, n_2_input, n_3_input)
            loss = torch.norm(X_Out * mask - X * mask, 2)
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            # if iter % 100 == 0:
            #     X_Out[mask == 1] = X[mask == 1]  
            #     X_Out = np.clip(X_Out.cpu().detach().numpy(), gt_min, gt_max)
            #     psnr_i, ssim_i, re_i = quality_index(gt.cpu().detach().numpy(), X_Out)
            #     psnr_list.append(psnr_i)
            #     # ssim_list.append(ssim_i)
            #     # re_list.append(re_i)

        Run_time = pbar.format_dict['elapsed']
        X_Out[mask == 1] = X[mask == 1]  
        X_Out = np.clip(X_Out.cpu().detach().numpy(), gt_min, gt_max)
        psnr, ssim, re = quality_index(gt.cpu().detach().numpy(), X_Out)

   
        save_name = f"{obs_name}_CB{idx}.mat"
        scipy.io.savemat('./results/' + save_name, {'X_Out': X_Out})

        output_line = f"CB: {idx}, W_decay={w_decay}, omega={omega}, shrink={shrink}, expand={expand}, iter={max_iter}, lr={lr_real}, PSNR={psnr:.4f}, SSIM={ssim:.4f}, RE={re:.4f}, Time={Run_time:.2f}s"
        with open('save_result.txt', 'a') as f:
            print(output_line, file=f)

        # plt.figure(dpi=300)
        # # plt.subplot(131)
        # plt.plot(range(1, len(psnr_list) + 1), psnr_list)
        # plt.title('PSNR')
        #
        # # plt.subplot(132)
        # # plt.plot(range(1, len(ssim_list) + 1), ssim_list)
        # # plt.title('SSIM')
        # #
        # # plt.subplot(133)
        # # plt.plot(range(1, len(re_list) + 1), re_list)
        # # plt.title('RE')
        #
        # plt.savefig('Evaluation_index.png')
        # plt.close()
