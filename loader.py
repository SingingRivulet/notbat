from cmath import tanh
import numpy
import torch
import torch.nn as nn
import collections
import ctypes
import random

pymidirenderer = ctypes.cdll.LoadLibrary("render-build/libpymidirender.so")


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


def loadMidi(path, sf, callback, section=4, sectionShift=0):
    def cb(a, b, c):
        buf_b = []
        buf_c = []
        for i in range(0, 512):
            buf_b.append(b[i])
            buf_c.append(c[i])
        callback(a, buf_b, buf_c)
    callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int))
    pymidirenderer.midirender_render(ctypes.c_char_p(path.encode()),
                                     ctypes.c_char_p(sf.encode()),
                                     ctypes.c_int(44100),
                                     ctypes.c_int(8192),
                                     ctypes.c_int(section),
                                     ctypes.c_int(sectionShift),
                                     ctypes.c_int(16),
                                     ctypes.c_int(16),
                                     ctypes.c_float(2),
                                     callback_type(cb))


def getDatasetList():
    res = []
    with open("datas/midi/process.sh") as f:
        line = f.readline()
        while line:
            arr = line.strip().split(" ")
            if len(arr) > 2:
                res.append(arr[2:])
            line = f.readline()
    random.shuffle(res)
    return res


def loadMidi_norm(path, sf, callback, section=4, sectionShift=0):
    with open(path+"/freqmax.txt") as out:
        maxv = float(out.readline())
        print("maxv:", maxv)

    def cb(count, input, output):
        tmp = []
        for n in input:
            tmp.append(n/maxv)
        callback(tmp, output)

    loadMidi(path+"/file.mid", sf, cb, section, sectionShift)


def loadMidi_tensor(path, sf, callback, section=4, sectionShift=0):  # 加载为张量
    que = collections.deque(maxlen=512)  # 用队列保存，提高性能
    count = [0]  # 统计数量

    def cb(input, output):
        que.append((input, output))
        if len(que) >= 512:
            if count[0] % 256 == 0:
                # 构建tensor
                tensor_in = []
                tensor_out = []
                for n in que:
                    tensor_in.append(n[0])
                    tensor_out.append(n[1])
                callback(
                    torch.tensor([[tensor_in]]),
                    torch.tensor([[tensor_out]]))
            count[0] += 1
    loadMidi_norm(path, sf, cb, section, sectionShift)


def loadAllDataset(callback):
    arr = getDatasetList()
    for line in arr:
        loadMidi_tensor(line[0], "datas/sndfnt.sf2", callback,
                        section=int(line[1]),
                        sectionShift=int(line[2]))


if __name__ == '__main__':
    # print(getDatasetList())
    def callback(a, b):
        print(a.size(), b.size())
    loadAllDataset(callback)
    #id = 0
    # for i in loadTable_tensor("./render-build/out.txt"):
    #    print(i[0].size(), i[1].size())
    #    #numpy.savetxt("test/"+str(id)+".txt", numpy.array(i[1][0][0]))
    #    id += 1
