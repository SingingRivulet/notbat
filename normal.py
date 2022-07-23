import loader
import os
# 归一化


def getMidiFreqMax(path):
    maxv = [0]

    if os.path.exists(path+"/freqmax.txt"):
        print(path, "已存在")
        return 0

    def callback(a, b, c):
        m = max(b)
        if m > maxv[0]:
            maxv[0] = m

    loader.loadMidi(path+"/file.mid",
                    "datas/sndfnt.sf2", callback)

    with open(path+"/freqmax.txt", "w") as out:
        out.write(str(maxv[0]))

    return maxv[0]


if __name__ == '__main__':
    arr = loader.getDatasetList()
    for line in arr:
        print(getMidiFreqMax(line[0]))
