# --coding:utf-8--
import torch
import matplotlib.pyplot as plt
from utils import bbox_to_rect,box_corner_to_center

#@save
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    in_height_tensor = torch.tensor(in_height, device=device)
    in_width_tensor = torch.tensor(in_width, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
    #                sizes[0] * torch.sqrt(ratio_tensor[1:])))\
    #                * in_height / in_width  # 处理矩形输入
    # h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
    #                sizes[0] / torch.sqrt(ratio_tensor[1:])))
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * torch.sqrt(in_height_tensor / in_width_tensor)  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:]))) * torch.sqrt(in_width_tensor / in_height_tensor)
    # print(w)
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

#@save
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)

    # print(jaccard.shape) num_anchors*num_gt_boxes
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1) # 针对每个anchor找最大的交并比, 每个anchor与哪个gt交并比大，返回索引
    # print(max_ious >= iou_threshold)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)   # 找最大IoU
        box_idx = (max_idx % num_gt_boxes).long()  # 得到具体的索引
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard  # 列填充 -1
        jaccard[anc_idx, :] = row_discard  # 行填充 -1
    return anchors_bbox_map

#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)   # 锚框转为中心宽高格式
    c_assigned_bb = box_corner_to_center(assigned_bb)   # 分配的边界框（真值）转为中心宽高格式
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]  # xy
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1   # 类别 +1为背景类
        assigned_bb[indices_true] = label[bb_idx, 1:]  # 坐标
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask   # 获取锚框与边界框的偏移量
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)  # 锚框偏移量， 掩码（哪些锚框是配对了的），类别标签

# img = plt.imread("F:\code\pytorch\Detect\datasets\data\img.png")
# h, w = img.shape[:2]
# # #
# # print(h, w)
# X = torch.rand(size=(1, 3, h, w))
# Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# # print(Y[0][0])
# # #
# # # fig = plt.imshow(img)
# # # fig.axes.add_patch(bbox_to_rect(Y[0][0], 'blue'))
# # # plt.show()
# boxes = Y.reshape(h, w, 5, 4)
# anchors = boxes[250, 250, :, :]
# gt = torch.tensor([[253/w, 118/h, 425/w, 310/h],[153/w, 18/h, 325/w, 210/h]])
# abmap = assign_anchor_to_bbox(gt, anchors=anchors, device='cpu')
# print(abmap)
# # # torch.set_figsize()
# bbox_scale = torch.tensor((w, h, w, h))
# fig = plt.imshow(img)
# # show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
# #             ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
# #              's=0.75, r=0.5'])
# index = torch.nonzero(abmap+1)
# print(index)
# showbox0 = boxes[250, 250, index[0][0], :]*bbox_scale
# showbox1 = boxes[250, 250, index[1][0], :]*bbox_scale
#
# fig.axes.add_patch(bbox_to_rect(showbox0, 'blue'))
# fig.axes.add_patch(bbox_to_rect(showbox1, 'pink'))
# fig.axes.add_patch(bbox_to_rect([253, 118, 425, 310], 'red'))
# fig.axes.add_patch(bbox_to_rect([153, 18, 325, 210], 'green'))
# # show_bboxes(fig.axes, torch.cat([boxes[250, 250, index, :] * bbox_scale, torch.tensor([253, 118, 425, 310],)], 1), ['anchor', 'gt'])
# plt.show()