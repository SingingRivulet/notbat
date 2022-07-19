from cmath import tanh
import numpy
import torch
import torch.nn as nn
import collections


def loadTable_frames(path):  # 加载单个表格文件
    with open(path) as f:
        line = f.readline()
        while line:
            arr = line.strip().split(",")
            if len(arr) > 1:
                freq_frame = []
                lab_frame = []
                for i in range(1, len(arr)):
                    pair = arr[i].split(":")
                    freq_frame.append(float(pair[0]))
                    lab_frame.append(int(pair[1]))
                yield (freq_frame, lab_frame)
            line = f.readline()


def loadTable_frames_norm(path):  # 归一化
    # 先获取最大
    max_n = 0
    for data in loadTable_frames(path):
        n = max(data[0])
        if n > max_n:
            max_n = n
    if max_n > 0:
        for data in loadTable_frames(path):
            tmp = []
            for n in data[0]:
                tmp.append(n/max_n)
            yield(tmp, data[1])


def loadTable_tensor(path):  # 加载为张量
    que = collections.deque(maxlen=512)  # 用队列保存，提高性能
    count = 0  # 统计数量
    for i in loadTable_frames_norm(path):
        que.append(i)
        if len(que) >= 512:
            if count % 256 == 0:
                # 构建tensor
                tensor_in = []
                tensor_out = []
                for n in que:
                    tensor_in.append(n[0])
                    tensor_out.append(n[1])

                yield(
                    torch.tensor([[tensor_in]]),
                    torch.tensor([[tensor_out]]))
            count += 1


if __name__ == '__main__':
    id = 0
    for i in loadTable_tensor("./render-build/out.txt"):
        print(i[0].size(), i[1].size())
        #numpy.savetxt("test/"+str(id)+".txt", numpy.array(i[1][0][0]))
        id += 1
