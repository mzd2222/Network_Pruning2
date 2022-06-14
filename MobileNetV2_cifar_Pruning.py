from models.MobileNetV2_cifar_Model import *

from utils.Data_loader import Data_Loader_CIFAR
from utils.Functions import *

import argparse

# ---------------------------------------
activations = []
record_activations = []


# 计算mask使用的activation_hook
def mask_activation_hook(module, input, output):
    global activations
    activations.append(output.clone().detach().cpu())
    return


# 记录前向推理使用的activation_hook
def forward_activation_hook(module, input, output):
    global record_activations
    record = input[0].clone().detach().cpu()
    record_activations.append(record)
    return


def Compute_layer_mask(imgs, model, percent, device, activation_func):
    """
    :argument 根据输入图片计算masks
    :param percent: 保留的比例
    :argument 根据输入图片获取模型的mask
    :param imgs: 输入图片tensor
    :param model:
    :param activation_func:
    :return: masks 维度为 [layer_num, c]
    """

    percent = 1 - percent  # 通过保留比例 计算出需要剪掉的比例percent

    global activations
    # 此处需要把模型更改为eval状态，否则在计算layer_mask时输入的数据会改变bn层参数，导致正确率下降
    model.eval()

    with torch.no_grad():
        imgs_masks = []
        hooks = []
        activations.clear()

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_hook(mask_activation_hook)
                hooks.append(hook)

        imgs = imgs.to(device)
        _ = model(imgs)

        for hook in hooks:
            hook.remove()

        # ------版本2 一层一层处理
        masks = []
        score_num_list = []
        for layer_activations in activations:
            if activation_func is not None:
                layer_activations = activation_func(layer_activations)
            # [img_num, c, h, w] => [img_num, c] --- [800, 64, 32, 32] => [800, 64]
            layer_activations_score = layer_activations.norm(dim=(2, 3), p=2)
            # [img_num, c]  eg [800, 64]
            layer_masks = torch.empty_like(layer_activations_score, dtype=torch.bool)
            # [image_num, c] 计算每一张图片的mask
            for idx, imgs_activations_score in enumerate(layer_activations_score):
                # [c]
                sorted_tensor, index = torch.sort(imgs_activations_score)
                threshold_index = int(len(sorted_tensor) * percent)
                threshold = sorted_tensor[threshold_index]
                one_img_mask = imgs_activations_score.gt(threshold)
                layer_masks[idx] = one_img_mask

            # 2.1 使用或
            one_layer_mask = layer_masks[0]
            # [img_num, c] => [c]  [800, 64] => [64]
            for img in layer_masks[1:]:
                for channel_id, channel_mask in enumerate(img):
                    one_layer_mask[channel_id] = one_layer_mask[channel_id] | channel_mask

            # 2.2 统计true数量
            # [c]  [64]
            # score_num = torch.sum(layer_masks, dim=0)
            # score_num_list.append(score_num)
            # sorted_tensor, _ = torch.sort(score_num)
            # score_threshold_index = int(len(sorted_tensor) * percent)
            # score_threshold = sorted_tensor[score_threshold_index]
            # one_layer_mask = score_num.gt(score_threshold)

            masks.append(one_layer_mask)

        return masks, score_num_list


def pre_processing_Pruning(model: nn.Module, masks, jump_layers):
    """

    :argument: 根据输入的mask，计算生成新模型所需的cfg，以及对应的新的layer_mask
              （和原本mask比其实知识把前两层全部置为1，前两层不剪枝）
    :param model:输入的预剪枝模型
    :param masks:剪枝用到的mask
    :param jump_layers: 剪枝跳过几层
    :return:
    """
    model.eval()
    cfg = []  # 新的网络结构参数
    count = 0  # bn层计数
    cfg_mask = []  # 计算新的mask
    pruned = 0  # 计算剪掉的通道数
    total = 0  # 总通道数
    cfg_old = []
    cfg_old2 = []

    for ii in masks:
        cfg_old.append(len(ii))
        cfg_old2.append(int(torch.sum(ii)))

    for index, module in enumerate(model.modules()):

        if isinstance(module, nn.BatchNorm2d):


            mask = masks[count]

            # 前两层不剪枝
            if count in jump_layers:
                mask = mask | True

            # 中间通道较少地方不剪枝
            if (count + 1) % 3 == 0:
                mask = mask | True

            # 对齐第二层和第三层通道
            if count % 3 == 0 and count+1 < len(masks):
                change_num = int(torch.sum(masks[count])) - int(torch.sum(masks[count + 1]))

                # if change_num > 0 : 前面true多 改count+1 改false为true
                # if change_num < 0 : 后面true多 改count   改false为true
                i = 0
                while change_num != 0:
                    if change_num > 0:
                        # 前面true多 改count+1 改false为true
                        if masks[count+1][i].item() is False:
                            masks[count+1][i] = torch.tensor(True, dtype=torch.bool)
                            change_num -= 1
                    if change_num < 0:
                        # 后面true多 改count   改false为true
                        if masks[count][i].item() is False:
                            masks[count][i] = torch.tensor(True, dtype=torch.bool)
                            change_num += 1
                    i += 1

            # 处理一下通道剩余0的情况
            if torch.sum(mask) == 0:
                mask[0] = 1

            # 当前层剩余通道
            cfg.append(int(torch.sum(mask)))
            # 当前层对应的mask向量
            cfg_mask.append(mask.clone())

            # 总通道数
            total += len(mask)

            # 总数减去保留的数量=剪掉的通道数
            pruned += len(mask) - torch.sum(mask)

            count += 1

    pruned_ratio = pruned / total
    print(cfg)
    print(cfg_old)
    print(cfg_old2)

    return cfg, cfg_mask, pruned_ratio.detach().item()


# 通道对齐 返回处理后的mask
def change_mask(residual_mask, now_mask):
    change_num = int(torch.sum(now_mask)) - int(torch.sum(residual_mask))
    # print(change_num)
    # if change_num > 0 : 当前True比较多  改true为false
    # if change_num < 0 : 当前False比较多 改false为true
    i = 0
    while change_num != 0:
        if change_num > 0:
            # 改true为false
            if now_mask[i].item() is True:
                now_mask[i] = torch.tensor(False, dtype=torch.bool)
                change_num -= 1
        if change_num < 0:
            # 改false为true
            if now_mask[i].item() is False:
                now_mask[i] = torch.tensor(True, dtype=torch.bool)
                change_num += 1
        i += 1

    if int(torch.sum(now_mask)) != int(torch.sum(residual_mask)):
        raise ValueError("通道没有对齐")

    return now_mask


# def Real_Pruning(model: nn.Module, cfg_masks, reserved_class):
#     """
#     :argument 根据cfg_mask即每个bn层的mask，将原始模型的参数拷贝至新模型，同时调整新模型的cs层和linear层
#               每个cs层通过设置index来实现剪枝
#
#     :param model:
#     :param cfg_masks:
#     :param reserved_class: 保留下的类
#     :return:返回剪枝后，拷贝完参数的模型，多余的类被剪掉
#     """
#
#     model.eval()
#
#     conv_idx = 0  # conv计数
#     bn_idx = 0  # bn计数 也是mask id计数
#     use_connect_sign = False
#
#     for idx, (name, module) in enumerate(model.named_modules()):
#
#         # 只负责剪第一层
#         if isinstance(module, nn.Conv2d):
#             if conv_idx == 0:
#                 module.weight.data = module.weight.data.clone()[cfg_masks[bn_idx]]
#                 module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                 conv_idx += 1
#         if isinstance(module, nn.BatchNorm2d):
#             if bn_idx == 0:
#                 module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 module.weight.data = module.weight.data.clone()[cfg_masks[bn_idx]]
#                 module.bias.data = module.bias.data.clone()[cfg_masks[bn_idx]]
#                 module.running_mean = module.running_mean.clone()[cfg_masks[bn_idx]]
#                 module.running_var = module.running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#         # 负责剪InvertedResidual
#         if isinstance(module, InvertedResidual):
#             if module.use_res_connect:
#
#                 # 0
#                 # 0.0 conv
#                 if use_connect_sign:
#
#                     now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx - 1])
#
#                     conv_weigh = module.conv[0][0].weight.data.clone()[:, now_mask, :, :]
#                     module.conv[0][0].in_channels = int(torch.sum(now_mask).item())
#                     use_connect_sign = False
#                 else:
#                     conv_weigh = module.conv[0][0].weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                     module.conv[0][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#                     residual_save_mask = cfg_masks[bn_idx - 1]
#
#                 module.conv[0][0].weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
#                 module.conv[0][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                 conv_idx += 1
#                 # 0.1 bn
#                 module.conv[0][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 module.conv[0][1].weight.data = module.conv[0][1].weight.data.clone()[cfg_masks[bn_idx]]
#                 module.conv[0][1].bias.data = module.conv[0][1].bias.data.clone()[cfg_masks[bn_idx]]
#                 module.conv[0][1].running_mean = module.conv[0][1].running_mean.clone()[cfg_masks[bn_idx]]
#                 module.conv[0][1].running_var = module.conv[0][1].running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#                 # 1
#                 # 1.0 conv
#                 # conv_weigh = module.conv[1][0].weight.data.clone()[:, cfg_masks[bn_idx-1], :, :]
#                 module.conv[1][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#                 module.conv[1][0].groups = module.conv[1][0].in_channels
#                 module.conv[1][0].weight.data = module.conv[1][0].weight.data.clone()[cfg_masks[bn_idx]]
#                 module.conv[1][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                 conv_idx += 1
#                 # 1.1 bn
#                 module.conv[1][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 module.conv[1][1].weight.data = module.conv[1][1].weight.data.clone()[cfg_masks[bn_idx]]
#                 module.conv[1][1].bias.data = module.conv[1][1].bias.data.clone()[cfg_masks[bn_idx]]
#                 module.conv[1][1].running_mean = module.conv[1][1].running_mean.clone()[cfg_masks[bn_idx]]
#                 module.conv[1][1].running_var = module.conv[1][1].running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#                 # 2
#                 # 2.0 conv
#                 now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx])
#
#                 conv_weigh = module.conv[2].weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                 module.conv[2].weight.data = conv_weigh.clone()[now_mask]
#                 module.conv[2].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#                 module.conv[2].out_channels = int(torch.sum(now_mask).item())
#                 conv_idx += 1
#                 # 2.1 bn
#                 now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx])
#
#                 module.conv[3].num_features = int(torch.sum(now_mask).item())
#                 module.conv[3].weight.data = module.conv[3].weight.data.clone()[now_mask]
#                 module.conv[3].bias.data = module.conv[3].bias.data.clone()[now_mask]
#                 module.conv[3].running_mean = module.conv[3].running_mean.clone()[now_mask]
#                 module.conv[3].running_var = module.conv[3].running_var.clone()[now_mask]
#                 bn_idx += 1
#
#                 use_connect_sign = True
#
#             else:
#                 for residual_module in module.modules():
#                     if isinstance(residual_module, nn.Conv2d):
#                         if use_connect_sign:
#                             now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx - 1])
#
#                             conv_weigh = residual_module.weight.data.clone()[:, now_mask, :, :]
#                             residual_module.in_channels = int(torch.sum(now_mask).item())
#                             use_connect_sign = False
#                         else:
#                             if residual_module.groups != 1:
#                                 conv_weigh = residual_module.weight.data.clone()
#                             else:
#                                 conv_weigh = residual_module.weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                             residual_module.in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#
#                         residual_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
#                         residual_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                         if residual_module.groups != 1:
#                             residual_module.groups = residual_module.in_channels
#                         conv_idx += 1
#
#                     if isinstance(residual_module, nn.BatchNorm2d):
#                         residual_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                         residual_module.weight.data = residual_module.weight.data.clone()[cfg_masks[bn_idx]]
#                         residual_module.bias.data = residual_module.bias.data.clone()[cfg_masks[bn_idx]]
#                         residual_module.running_mean = residual_module.running_mean.clone()[cfg_masks[bn_idx]]
#                         residual_module.running_var = residual_module.running_var.clone()[cfg_masks[bn_idx]]
#                         bn_idx += 1
#
#         # 只负责剪最后一层
#         if isinstance(module, nn.Conv2d):
#             if name == 'features.18.0':
#                 if use_connect_sign:
#
#                     now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx - 1])
#
#                     conv_weigh = module.weight.data.clone()[:, now_mask, :, :]
#                     module.in_channels = int(torch.sum(now_mask).item())
#                     use_connect_sign = False
#                 else:
#                     conv_weigh = module.weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                     module.in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#
#                 module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
#                 module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#         if isinstance(module, nn.BatchNorm2d):
#             if name == 'features.18.1':
#                 module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 module.weight.data = module.weight.data.clone()[cfg_masks[bn_idx]]
#                 module.bias.data = module.bias.data.clone()[cfg_masks[bn_idx]]
#                 module.running_mean = module.running_mean.clone()[cfg_masks[bn_idx]]
#                 module.running_var = module.running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#         if isinstance(module, nn.Linear):
#
#             out_mask = torch.full([module.weight.data.size(0)], False)
#             for ii in reserved_class:
#                 out_mask[ii] = True
#
#             # 改变输入size
#             fc_data = module.weight.data.clone()[:, cfg_masks[-1]]
#             # 改变输出size
#             fc_data = fc_data.clone()[out_mask, :]
#
#             module.weight.data = fc_data.clone()
#             module.bias.data = module.bias.data.clone()[out_mask]
#             module.in_features = int(torch.sum(cfg_masks[-1]).item())
#             module.out_features = int(torch.sum(out_mask).item())
#     return model


# def Real_Pruning(new_model: nn.Module, old_model: nn.Module, cfg_masks, reserved_class):
#     """
#     :argument 根据cfg_mask即每个bn层的mask，将原始模型的参数拷贝至新模型，同时调整新模型的cs层和linear层
#               每个cs层通过设置index来实现剪枝
#
#     :param model:
#     :param cfg_masks:
#     :param reserved_class: 保留下的类
#     :return:返回剪枝后，拷贝完参数的模型，多余的类被剪掉
#     """
#
#     new_model.eval()
#     old_model.eval()
#
#     conv_idx = 0  # conv计数
#     bn_idx = 0  # bn计数 也是mask id计数
#     use_connect_sign = False
#
#     for idx, (new_name_module, old_name_module) in enumerate(zip(new_model.named_modules(), old_model.named_modules())):
#
#         new_name = new_name_module[0]
#         new_module = new_name_module[1]
#         old_name = old_name_module[0]
#         old_module = old_name_module[1]
#
#         # 只负责剪第一层
#         if isinstance(old_module, nn.Conv2d):
#             if conv_idx == 0:
#                 new_module.weight.data = old_module.weight.data.clone()[cfg_masks[bn_idx]]
#                 new_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                 conv_idx += 1
#         if isinstance(old_module, nn.BatchNorm2d):
#             if bn_idx == 0:
#                 new_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 new_module.weight.data = old_module.weight.data.clone()[cfg_masks[bn_idx]]
#                 new_module.bias.data = old_module.bias.data.clone()[cfg_masks[bn_idx]]
#                 new_module.running_mean = old_module.running_mean.clone()[cfg_masks[bn_idx]]
#                 new_module.running_var = old_module.running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#         # 负责剪InvertedResidual
#         if isinstance(old_module, InvertedResidual):
#             if old_module.use_res_connect:
#
#                 # 0
#                 # 0.0 conv
#                 if use_connect_sign:
#
#                     now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx - 1])
#
#                     conv_weigh = old_module.conv[0][0].weight.data.clone()[:, now_mask, :, :]
#                     new_module.conv[0][0].in_channels = int(torch.sum(now_mask).item())
#                     use_connect_sign = False
#                 else:
#                     conv_weigh = old_module.conv[0][0].weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                     new_module.conv[0][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#                     residual_save_mask = cfg_masks[bn_idx - 1]
#
#                 new_module.conv[0][0].weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[0][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                 conv_idx += 1
#                 # 0.1 bn
#                 new_module.conv[0][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 new_module.conv[0][1].weight.data = old_module.conv[0][1].weight.data.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[0][1].bias.data = old_module.conv[0][1].bias.data.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[0][1].running_mean = old_module.conv[0][1].running_mean.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[0][1].running_var = old_module.conv[0][1].running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#                 # 1
#                 # 1.0 conv
#                 # conv_weigh = module.conv[1][0].weight.data.clone()[:, cfg_masks[bn_idx-1], :, :]
#                 new_module.conv[1][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#                 new_module.conv[1][0].groups = new_module.conv[1][0].in_channels
#                 new_module.conv[1][0].weight.data = old_module.conv[1][0].weight.data.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[1][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                 conv_idx += 1
#                 # 1.1 bn
#                 new_module.conv[1][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 new_module.conv[1][1].weight.data = old_module.conv[1][1].weight.data.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[1][1].bias.data = old_module.conv[1][1].bias.data.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[1][1].running_mean = old_module.conv[1][1].running_mean.clone()[cfg_masks[bn_idx]]
#                 new_module.conv[1][1].running_var = old_module.conv[1][1].running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#                 # 2
#                 # 2.0 conv
#                 now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx])
#
#                 conv_weigh = old_module.conv[2].weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                 new_module.conv[2].weight.data = conv_weigh.clone()[now_mask]
#                 new_module.conv[2].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#                 new_module.conv[2].out_channels = int(torch.sum(now_mask).item())
#                 conv_idx += 1
#                 # 2.1 bn
#                 now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx])
#
#                 new_module.conv[3].num_features = int(torch.sum(now_mask).item())
#                 new_module.conv[3].weight.data = old_module.conv[3].weight.data.clone()[now_mask]
#                 new_module.conv[3].bias.data = old_module.conv[3].bias.data.clone()[now_mask]
#                 new_module.conv[3].running_mean = old_module.conv[3].running_mean.clone()[now_mask]
#                 new_module.conv[3].running_var = old_module.conv[3].running_var.clone()[now_mask]
#                 bn_idx += 1
#
#                 use_connect_sign = True
#
#             else:
#                 for new_residual_module, old_residual_module in zip(new_module.modules(), old_module.modules()):
#                     if isinstance(old_residual_module, nn.Conv2d):
#                         if use_connect_sign:
#                             now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx - 1])
#
#                             conv_weigh = old_residual_module.weight.data.clone()[:, now_mask, :, :]
#                             new_residual_module.in_channels = int(torch.sum(now_mask).item())
#                             use_connect_sign = False
#                         else:
#                             if new_residual_module.groups != 1:
#                                 conv_weigh = old_residual_module.weight.data.clone()
#                             else:
#                                 conv_weigh = old_residual_module.weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                             new_residual_module.in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#
#                         new_residual_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
#                         new_residual_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#                         if new_residual_module.groups != 1:
#                             new_residual_module.groups = new_residual_module.in_channels
#                         conv_idx += 1
#
#                     if isinstance(old_residual_module, nn.BatchNorm2d):
#                         new_residual_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                         new_residual_module.weight.data = old_residual_module.weight.data.clone()[cfg_masks[bn_idx]]
#                         new_residual_module.bias.data = old_residual_module.bias.data.clone()[cfg_masks[bn_idx]]
#                         new_residual_module.running_mean = old_residual_module.running_mean.clone()[cfg_masks[bn_idx]]
#                         new_residual_module.running_var = old_residual_module.running_var.clone()[cfg_masks[bn_idx]]
#                         bn_idx += 1
#
#         # 只负责剪最后一层
#         if isinstance(old_module, nn.Conv2d):
#             if old_name == 'features.18.0':
#                 if use_connect_sign:
#
#                     now_mask = change_mask(residual_save_mask, cfg_masks[bn_idx - 1])
#
#                     conv_weigh = old_module.weight.data.clone()[:, now_mask, :, :]
#                     new_module.in_channels = int(torch.sum(now_mask).item())
#                     use_connect_sign = False
#                 else:
#                     conv_weigh = old_module.weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
#                     new_module.in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
#
#                 new_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
#                 new_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
#         if isinstance(old_module, nn.BatchNorm2d):
#             if old_name == 'features.18.1':
#                 new_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
#                 new_module.weight.data = old_module.weight.data.clone()[cfg_masks[bn_idx]]
#                 new_module.bias.data = old_module.bias.data.clone()[cfg_masks[bn_idx]]
#                 new_module.running_mean = old_module.running_mean.clone()[cfg_masks[bn_idx]]
#                 new_module.running_var = old_module.running_var.clone()[cfg_masks[bn_idx]]
#                 bn_idx += 1
#
#         if isinstance(old_module, nn.Linear):
#
#             out_mask = torch.full([old_module.weight.data.size(0)], False)
#             for ii in reserved_class:
#                 out_mask[ii] = True
#
#             # 改变输入size
#             fc_data = old_module.weight.data.clone()[:, cfg_masks[-1]]
#             # 改变输出size
#             fc_data = fc_data.clone()[out_mask, :]
#
#             new_module.weight.data = fc_data.clone()
#             new_module.bias.data = old_module.bias.data.clone()[out_mask]
#             new_module.in_features = int(torch.sum(cfg_masks[-1]).item())
#             new_module.out_features = int(torch.sum(out_mask).item())
#
#     return new_model


def Real_Pruning(new_model: nn.Module, old_model: nn.Module, cfg_masks, reserved_class):
    """
    :argument 根据cfg_mask即每个bn层的mask，将原始模型的参数拷贝至新模型，同时调整新模型的cs层和linear层
              每个cs层通过设置index来实现剪枝

    :param model:
    :param cfg_masks:
    :param reserved_class: 保留下的类
    :return:返回剪枝后，拷贝完参数的模型，多余的类被剪掉
    """

    new_model.eval()
    old_model.eval()

    conv_idx = 0  # conv计数
    bn_idx = 0  # bn计数 也是mask id计数
    use_connect_sign = False

    for idx, (new_name_module, old_name_module) in enumerate(zip(new_model.named_modules(), old_model.named_modules())):

        new_name = new_name_module[0]
        new_module = new_name_module[1]
        old_name = old_name_module[0]
        old_module = old_name_module[1]

        # 只负责剪第一层
        if isinstance(old_module, nn.Conv2d):
            if conv_idx == 0:
                new_module.weight.data = old_module.weight.data.clone()
                conv_idx += 1
        if isinstance(old_module, nn.BatchNorm2d):
            if bn_idx == 0:
                new_module.weight.data = old_module.weight.data.clone()
                new_module.bias.data = old_module.bias.data.clone()
                new_module.running_mean = old_module.running_mean.clone()
                new_module.running_var = old_module.running_var.clone()
                bn_idx += 1

        # 负责剪InvertedResidual
        if isinstance(old_module, InvertedResidual):
            if old_module.use_res_connect:

                # 0
                # 0.0 conv
                conv_weigh = old_module.conv[0][0].weight.data.clone()
                # new_module.conv[0][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
                new_module.conv[0][0].weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
                # new_module.conv[0][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
                conv_idx += 1
                # 0.1 bn
                new_module.conv[0][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                new_module.conv[0][1].weight.data = old_module.conv[0][1].weight.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[0][1].bias.data = old_module.conv[0][1].bias.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[0][1].running_mean = old_module.conv[0][1].running_mean.clone()[cfg_masks[bn_idx]]
                new_module.conv[0][1].running_var = old_module.conv[0][1].running_var.clone()[cfg_masks[bn_idx]]
                bn_idx += 1

                # 1
                # 1.0 conv
                # conv_weigh = module.conv[1][0].weight.data.clone()[:, cfg_masks[bn_idx-1], :, :]
                new_module.conv[1][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
                new_module.conv[1][0].groups = new_module.conv[1][0].in_channels
                new_module.conv[1][0].weight.data = old_module.conv[1][0].weight.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
                conv_idx += 1
                # 1.1 bn
                new_module.conv[1][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                new_module.conv[1][1].weight.data = old_module.conv[1][1].weight.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][1].bias.data = old_module.conv[1][1].bias.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][1].running_mean = old_module.conv[1][1].running_mean.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][1].running_var = old_module.conv[1][1].running_var.clone()[cfg_masks[bn_idx]]
                bn_idx += 1

                # 2
                # 2.0 conv
                conv_weigh = old_module.conv[2].weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
                new_module.conv[2].weight.data = conv_weigh.clone()
                new_module.conv[2].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())

                conv_idx += 1
                # 2.1 bn
                new_module.conv[3].weight.data = old_module.conv[3].weight.data.clone()
                new_module.conv[3].bias.data = old_module.conv[3].bias.data.clone()
                new_module.conv[3].running_mean = old_module.conv[3].running_mean.clone()
                new_module.conv[3].running_var = old_module.conv[3].running_var.clone()
                bn_idx += 1

            else:
                for new_residual_module, old_residual_module in zip(new_module.modules(), old_module.modules()):
                    if isinstance(old_residual_module, nn.Conv2d):

                        if new_residual_module.groups != 1:
                            conv_weigh = old_residual_module.weight.data.clone()
                        else:
                            conv_weigh = old_residual_module.weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
                        new_residual_module.in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())

                        new_residual_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
                        new_residual_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
                        if new_residual_module.groups != 1:
                            new_residual_module.groups = new_residual_module.in_channels
                        conv_idx += 1

                    if isinstance(old_residual_module, nn.BatchNorm2d):
                        new_residual_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                        new_residual_module.weight.data = old_residual_module.weight.data.clone()[cfg_masks[bn_idx]]
                        new_residual_module.bias.data = old_residual_module.bias.data.clone()[cfg_masks[bn_idx]]
                        new_residual_module.running_mean = old_residual_module.running_mean.clone()[cfg_masks[bn_idx]]
                        new_residual_module.running_var = old_residual_module.running_var.clone()[cfg_masks[bn_idx]]
                        bn_idx += 1

        # 只负责剪最后一层
        if isinstance(old_module, nn.Conv2d):
            if old_name == 'features.18.0':
                conv_weigh = old_module.weight.data.clone()
                new_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
                new_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())

        if isinstance(old_module, nn.BatchNorm2d):
            if old_name == 'features.18.1':
                new_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                new_module.weight.data = old_module.weight.data.clone()[cfg_masks[bn_idx]]
                new_module.bias.data = old_module.bias.data.clone()[cfg_masks[bn_idx]]
                new_module.running_mean = old_module.running_mean.clone()[cfg_masks[bn_idx]]
                new_module.running_var = old_module.running_var.clone()[cfg_masks[bn_idx]]
                bn_idx += 1

        if isinstance(old_module, nn.Linear):

            out_mask = torch.full([old_module.weight.data.size(0)], False)
            for ii in reserved_class:
                out_mask[ii] = True

            # 改变输入size
            fc_data = old_module.weight.data.clone()[:, cfg_masks[-1]]
            # 改变输出size
            fc_data = fc_data.clone()[out_mask, :]

            new_module.weight.data = fc_data.clone()
            new_module.bias.data = old_module.bias.data.clone()[out_mask]
            new_module.in_features = int(torch.sum(cfg_masks[-1]).item())
            new_module.out_features = int(torch.sum(out_mask).item())

    return new_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-train_b", "--train_batch_size", help="train_batch_size", default=256)
    parser.add_argument("-test_b", "--test_batch_size", help="test_batch_size", default=256)

    parser.add_argument("-epoch1", "--fine_tuning_epoch1", help="fine_tuning_epoch1", default=50)
    parser.add_argument("-test_epoch", "--test_epoch", help="test_epoch", default=10)

    parser.add_argument("-fine_batch_size", "--fine_tuning_batch_size", help="fine_tuning_batch_size", default=64)

    parser.add_argument("-test_time", "--test_time", help="if test_time", default=True)
    parser.add_argument("-test_latency", "--test_latency", help="if test_latency", default=False)


    args = parser.parse_args()

    # dataSet_name = 'CIFAR100'
    dataSet_name = 'CIFAR10'

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(train_batch_size=int(args.train_batch_size),
                                    test_batch_size=int(args.test_batch_size),
                                    dataSet=dataSet_name, use_data_Augmentation=False,
                                    download=False, train_shuffle=True)
    model_path = './models/MobileNetV2/MovileNetV2_cifar_before_8842.pkl'
    # model_path = './models/MobileNetV2/MovileNetV2_cifar100_before_6249.pkl'

    model = torch.load(f=model_path).to(device)

    model.eval()

    # 测试剪枝前模型
    # _, _, class_acc = test_reserved_classes(model=model, device=device, reserved_classes=[iii for iii in range(100)],
    #                                         test_data_loader=data_loader.test_data_loader, is_print=True,
    #                                         test_class=True, dataset_num_class=data_loader.dataset_num_class)
    # acc_c = []
    # for c_i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #     acc_c.append(np.mean(class_acc[:c_i]))
    # print(acc_c)
    # exit(0)

    test_time = True

    # reserved_classes = [i for i in range(5)]

    version_id = 0  # 指定
    model_id = 0  # 保存模型的id

    if test_time:
        fine_tuning_epoch1 = int(args.fine_tuning_epoch1)
    else:
        fine_tuning_epoch = 20

    # manual_radio = 0.01
    # jump_layers = 6  # 跳过层数为3的倍数
    jump_layers_list = [i for i in range(52)]
    # jump_layers_list.remove(6)
    # jump_layers_list.remove(7)
    # jump_layers_list.remove(8)
    # jump_layers_list.remove(9)   # B3_1
    # jump_layers_list.remove(10)  # B3_1
    # jump_layers_list.remove(11)  # B3_1
    # jump_layers_list.remove(12)  # B3_2
    # jump_layers_list.remove(13)  # B3_2
    # jump_layers_list.remove(14)  # B3_2
    # jump_layers_list.remove(15)  # B3_3
    # jump_layers_list.remove(16)  # B3_3
    # jump_layers_list.remove(17)  # B3_3
    # jump_layers_list.remove(18)  # B3_1
    # jump_layers_list.remove(19)  # B3_1
    # jump_layers_list.remove(20)  # B3_1
    jump_layers_list.remove(21)  # B3_1
    jump_layers_list.remove(22)  # B3_2
    jump_layers_list.remove(23)  # B3_2
    jump_layers_list.remove(24)  # B3_2
    jump_layers_list.remove(25)  # B3_3
    jump_layers_list.remove(26)  # B3_3
    jump_layers_list.remove(27)  # B3_3
    jump_layers_list.remove(28)  # B3_3
    jump_layers_list.remove(29)  # B3_1
    jump_layers_list.remove(30)  # B3_1
    jump_layers_list.remove(31)  # B3_1
    jump_layers_list.remove(32)  # B3_2
    jump_layers_list.remove(33)  # B3_2
    jump_layers_list.remove(34)  # B3_2
    jump_layers_list.remove(35)  # B3_3
    jump_layers_list.remove(36)  # B3_3
    jump_layers_list.remove(37)  # B3_3
    jump_layers_list.remove(38)  # B3_3
    jump_layers_list.remove(39)  # B3_1
    jump_layers_list.remove(40)  # B3_1
    jump_layers_list.remove(41)  # B3_1
    jump_layers_list.remove(42)  # B3_2
    jump_layers_list.remove(43)  # B3_2
    jump_layers_list.remove(44)  # B3_2
    jump_layers_list.remove(45)  # B3_3
    jump_layers_list.remove(46)  # B3_3
    jump_layers_list.remove(47)  # B3_3
    jump_layers_list.remove(48)  # B3_3
    jump_layers_list.remove(49)  # B3_1
    # jump_layers_list.remove(50)  # B3_1
    # jump_layers_list.remove(51)  # B3_1
    # print(jump_layers_list)

    fine_tuning_lr = 0.001
    fine_tuning_batch_size = 128
    fine_tuning_pics_num = 256
    # fine_tuning_pics_num_list = [1, 4, 8, 16, 32, 64, 128, 150, 180, 200, 230, 256, 300]

    use_KL_divergence = True
    divide_radio = 1
    redundancy_num = 100

    record_imgs_num = 512
    record_batch = 128

    frozen = False

    version_msg = "版本备注:"

    msg_save_path = "./msg/latency_msg/mobile.txt"
    model_save_path = './models/MobileNetV2/version'
    # msg_save_path = "/kaggle/working/model_msg.txt"
    # model_save_path = '/kaggle/working/version'
    model_save_path += str(version_id) + '_MobileNetV2_after_model_' + str(model_id) + '.pkl'

    max_kc = None
    min_kc = None
    FLOPs_radio = 0.00
    Parameters_radio = 0.00
    # ----------------------------------------------------------------------
    # --------------
    # ----------------------------------------------------------------------

    if test_time:
        time_prune_list = []
        time_choose_list = []
        time_fine_tune_list = []

    reserved_classes_list = [[iii for iii in range(2)], [iii for iii in range(5)], [iii for iii in range(8)]]
    manual_radio_list = []

    for reserved_classes, manual_radio in zip(reserved_classes_list, manual_radio_list):

        activations.clear()
        record_activations.clear()

        # ----------第一步：进行一定数量的前向推理forward，并记录图片和中间激活值
        record_dataloader = read_Img_by_class(pics_num=record_imgs_num,
                                              batch_size=record_batch,
                                              target_class=reserved_classes,
                                              data_loader=data_loader.train_data_loader,
                                              shuffle=False)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(forward_activation_hook)

        # 模拟实际前向推理(剪枝前)，记录数据
        model.eval()
        with torch.no_grad():
            for forward_x, _ in record_dataloader:
                forward_x = forward_x.to(device)
                _ = model(forward_x)
        hook.remove()

        model.eval()
        old_FLOPs, old_parameters = cal_FLOPs_and_Parameters(model, device)
        # 测试不同的激活函数
        # [None, nn.ReLU(), nn.LeakyReLU(), F.relu6, nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()]

        # activations_list = [None, nn.ReLU(), nn.LeakyReLU()]
        # for i in range(0, 101, 5):
        #     # print(i / 100.0)
        #     activations_list.append(nn.LeakyReLU(negative_slope=i/100.0))
        # activations_list.extend([nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()])

        # activations_list = [nn.ReLU(), nn.ReLU()]
        # negative_slope_list = [0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.11, 0.12, 0.13, 0.14, 0.16, 0.18]
        # for negative_slope in negative_slope_list:
        #     activations_list.append(nn.LeakyReLU(negative_slope=negative_slope))

        for act_func in [nn.LeakyReLU(negative_slope=0.06)]:
            # ----------第二部：根据前向推理图片获取保留的类，并计算mask，进行剪枝，获得剪枝后的新模型

            for i in range(1):

                if test_time:
                    if i == 0:
                        fine_tuning_epoch = 1
                    else:
                        fine_tuning_epoch = fine_tuning_epoch1

                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

                # 从记录数据中选取一部分计算mask
                imgs_tensor = choose_mask_imgs(target_class=reserved_classes,
                                               data_loader=record_dataloader,
                                               pics_num=30)
                layer_masks, score_num_list = Compute_layer_mask(imgs=imgs_tensor, model=model, percent=manual_radio,
                                                                 device=device, activation_func=act_func)

                # for idx, score_num in enumerate(score_num_list):
                #     socre_num_numpy = score_num.numpy()
                #     np.save('./activ_numpy/MobileNetV2/layer' + str(idx + 1) + '.npy', socre_num_numpy)

                # 对比不同数量图片计算mask的作图
                # rv_pics_num_list = [1, 2, 5, 10, 20, 40, 80, 120]
                # for rv_pics_num in rv_pics_num_list:
                #     imgs_tensor = choose_mask_imgs(target_class=reserved_classes,
                #                                    data_loader=record_dataloader,
                #                                    pics_num=rv_pics_num)
                #     layer_masks, _ = Compute_layer_mask(imgs=imgs_tensor, model=model, percent=manual_radio, device=device,
                #                                         activation_func=act_func)
                #
                #     for idx, mask_ in enumerate(layer_masks):
                #         mask = mask_.clone().cpu().numpy().reshape(1, -1)
                #         try:
                #             old_mask = np.load('./mask_numpy/MobileNetV2/layer' + str(idx + 1) + '.npy')
                #             new_mask = np.vstack((old_mask, mask))
                #             print(new_mask.shape)
                #             np.save('./mask_numpy/MobileNetV2/layer' + str(idx + 1) + '.npy', new_mask)
                #         except:
                #             np.save('./mask_numpy/MobileNetV2/layer' + str(idx + 1) + '.npy', mask)
                # exit(0)

                # 预剪枝,计算mask 跳过层数不能为3的倍数
                cfg, cfg_masks, filter_remove_radio = pre_processing_Pruning(model, layer_masks, jump_layers=jump_layers_list)
                filter_remain_radio = 1 - filter_remove_radio

                redundancy_num_list = [100]
                for redundancy_num in redundancy_num_list:


                    # 正式剪枝,参数拷贝
                    new_model = torch.load(f=model_path).to(device)
                    model_after_pruning = Real_Pruning(new_model=new_model, old_model=model, cfg_masks=cfg_masks, reserved_class=reserved_classes)

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_prune = time_out_pr - time_in_pr
                            print("剪枝时间：" + str(time_prune) + "s")
                            time_prune_list.append(time_prune)


                    # print(model_after_pruning)
                    # 剪枝后模型测试
                    # model_after_pruning.eval()
                    # test_reserved_classes(model=model_after_pruning, device=device, reserved_classes=reserved_classes,
                    #                       test_data_loader=data_loader.test_data_loader, test_class=True, is_print=True,
                    #                       dataset_num_class=data_loader.dataset_num_class)

                    # 计算多种压缩率标准
                    model_after_pruning.eval()
                    new_FLOPs, new_parameters = cal_FLOPs_and_Parameters(model_after_pruning, device)
                    FLOPs_radio = new_FLOPs / old_FLOPs
                    Parameters_radio = new_parameters / old_parameters

                    # 剪枝后模型测试
                    # test_reserved_classes(model=model_after_pruning, device=device, reserved_classes=reserved_classes,
                    #                       test_data_loader=data_loader.test_data_loader, test_class=True, is_print=True)

                    # 多GPU微调
                    if torch.cuda.device_count() > 1:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        model_after_pruning = nn.DataParallel(model_after_pruning)
                    model_after_pruning.to(device)

                    print("model_id:" + str(model_id)
                          + " ---filter_remain_radio:" + str(filter_remain_radio)
                          + " ---FLOPs_radio:" + str(FLOPs_radio)
                          + " ---Parameters_radio:" + str(Parameters_radio)
                          + "  运行：")

                    # ----------第三步：从前向推理记录的图片中，使用算法选取一部分进行微调


                    # torch.save(model_after_pruning, './models/now_model.pkl')

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_in_pr = time.time()

                    # for fine_tuning_pics_num in [256]:
                    # model_after_pruning = torch.load(f='./models/now_model.pkl').to(device)

                    # --------------------------------------------- 微调
                    # 选取微调数据

                    # fine_tuning_loader = get_fine_tuning_data_loader1(reserved_classes,
                    #                                                   pics_num=fine_tuning_pics_num,
                    #                                                   batch_size=fine_tuning_batch_size,
                    #                                                   data_loader=record_dataloader,
                    #                                                   redundancy_num=redundancy_num,
                    #                                                   use_KL=False)

                    fine_tuning_loader, max_kc, min_kc = get_fine_tuning_data_loader2(record_activations,
                                                                                      reserved_classes,
                                                                                      pics_num=fine_tuning_pics_num,
                                                                                      batch_size=fine_tuning_batch_size,
                                                                                      data_loader=record_dataloader,
                                                                                      redundancy_num=redundancy_num,
                                                                                      divide_radio=divide_radio,
                                                                                      use_max=True)

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_choose = time_out_pr - time_in_pr
                            print("选数据时间：" + str(time_choose) + "s")
                            time_choose_list.append(time_choose)

                        if i != 0:
                            torch.cuda.synchronize()
                            time_in_pr = time.time()

                    # 微调
                    best_acc, acc_list, loss_list = fine_tuning(model_after_pruning, reserved_classes,
                                                                EPOCH=fine_tuning_epoch, lr=fine_tuning_lr,
                                                                device=device,
                                                                train_data_loader=fine_tuning_loader,
                                                                test_data_loader=data_loader.test_data_loader,
                                                                model_save_path=model_save_path,
                                                                use_all_data=False,
                                                                frozen=frozen,
                                                                dataset_num_class=data_loader.dataset_num_class)

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_fine_tune = time_out_pr - time_in_pr
                            print("微调时间：" + str(time_fine_tune) + "s")
                            time_fine_tune_list.append(time_fine_tune)

                    # 测试延迟
                    # model.eval()
                    # model_after_pruning.eval()
                    # input_size = (100, 3, 32, 32)
                    # latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=10, device='cuda')
                    # latency_old = compute_latency_ms_pytorch(model, input_size, iterations=10, device='cuda')
                    # latency_old = compute_latency_ms_pytorch(model, input_size, iterations=100, device='cuda')
                    # latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=100, device='cuda')
                    # print('model:{}, | latency: {}'.format('new', latency_new))
                    # print('model:{}, | latency: {}'.format('old', latency_old))


                    print("model_id:---" + str(model_id) +
                          " best_acc:----" + str(best_acc) +
                          " reserved_classes:---" + str(reserved_classes) +
                          " manual_radio:---" + str(manual_radio) +
                          " filter_remain_radio:---" + str(filter_remain_radio) +
                          '\n')

                    with open(msg_save_path, "a") as fp:
                        space = " "

                        fp.write(str(version_id) + space +
                                 str(model_id) + space +
                                 str(round(best_acc + 0.0001, 4)) + space +
                                 str(fine_tuning_batch_size) + space +
                                 str(fine_tuning_pics_num) + space +
                                 str(fine_tuning_epoch) + space +
                                 str(fine_tuning_lr) + space +
                                 str(redundancy_num) + space +
                                 str(divide_radio) + space +
                                 str(use_KL_divergence) + space +
                                 str(round(manual_radio, 4)) + space +
                                 str(round(FLOPs_radio, 4)) + space +
                                 str(round(Parameters_radio, 4)) + space +
                                 str(round(filter_remain_radio, 4)) + space +
                                 str(act_func) + space +
                                 str(reserved_classes) + space +
                                 str(max_kc) + space +
                                 str(min_kc) + space +
                                 version_msg + space +
                                 model_save_path + space +
                                 "\n")

                    model_id += 1

    with open(msg_save_path, "a") as fp:
        fp.write("\n")

    # 方法1
    # fine_tuning_loader = get_fine_tuning_data_loader1(reserved_classes,
    #                                                   pics_num=fine_tuning_pics_num,
    #                                                   batch_size=fine_tuning_batch_size,
    #                                                   data_loader=data_loader.train_data_loader,
    #                                                   redundancy_num=redundancy_num,
    #                                                   use_KL=False)

    # 方法3
    # fine_tuning_loader, max_kc, min_kc = get_fine_tuning_data_loader3(record_activations,
    #                                                                   reserved_classes,
    #                                                                   pics_num=fine_tuning_pics_num,
    #                                                                   batch_size=fine_tuning_batch_size,
    #                                                                   data_loader=record_dataloader,
    #                                                                   redundancy_num=redundancy_num,
    #                                                                   use_max=True)

    # 方法4
    # fine_tuning_loader, max_kc, min_kc = get_fine_tuning_data_loader4(record_activations,
    #                                                                   reserved_classes,
    #                                                                   pics_num=fine_tuning_pics_num,
    #                                                                   batch_size=fine_tuning_batch_size,
    #                                                                   data_loader=record_dataloader,
    #                                                                   redundancy_num=redundancy_num,
    #                                                                   use_max=True)

    # 画图
    # draw_acc_loss(acc_list=acc_list, loss_list=loss_list, line_id=model_id, vis=vis)
