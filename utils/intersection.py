import numpy as np


# 矩形aと、複数の矩形bのIoUを計算
def iou_np(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])

    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:, 2] - b[:, 0] + 1) \
             * (b[:, 3] - b[:, 1] + 1)

    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:, 0])  # xmin
    aby_mn = np.maximum(a[1], b[:, 1])  # ymin
    abx_mx = np.minimum(a[2], b[:, 2])  # xmax
    aby_mx = np.minimum(a[3], b[:, 3])  # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w * h

    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)
    return iou


def is_draw(bounding_boxes, scores, index):

    iou_s = iou_np(bounding_boxes[index], bounding_boxes)

    for i, iou in enumerate(iou_s):
        if (iou > 0) and (i != index) and (scores[index] < scores[i]):
            return False

    return True


def is_same_person(det, det_arr, thresh=0.8):
    """
    検出されたものが前フレームと同じかiouベースで判断を行う
    det : 今のフレームの検出結果 (numpy)
    det_arr : 今までの検出結果 (リスト)
    thresh : 同じ物体だと
    return : 新物体→-1 or 物体のID
    """
    # det_arrから前フレームの検出結果のみを抽出する
    pre_dets = np.zeros((len(det_arr), 4))
    for i in range(len(det_arr)):
        # print(pre_dets[:, i].shape)
        # print(det_arr[i][-1].shape)
        pre_dets[i, :] = det_arr[i][-1]

    iou_s = iou_np(det, pre_dets)

    for id, iou in enumerate(iou_s):
        if iou > thresh:
            return id

    return -1

