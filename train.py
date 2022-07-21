import model
import loader
import torch


class CrossEntropyLoss2d(torch.nn.Module):  # loss函数

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        # 分类需要softmax
        output_softmax = torch.log_softmax(outputs, dim=1)
        # print(output_softmax)
        output_target = targets[:, 0, :, :]
        return self.loss(output_softmax, output_target)


if __name__ == '__main__':
    print("准备训练")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备", device)
    m = model.PSPNet(3).to(device=device)  # 模型
    loss_function = CrossEntropyLoss2d(torch.ones(3))  # 三个分类
    optimizer = torch.optim.SGD(m.parameters(), lr=0.001)
    index = 0
    model_id = 0
    print("开始训练")
    for arr in loader.loadTable_tensor("datas/test.mid.txt"):
        m.zero_grad()
        # 准备数据
        data_in = arr[0].to(device=device)
        targets = arr[1].to(device=device)
        #print(data_in, targets)
        # 前向传播
        tag_scores = m(data_in)
        # 计算损失
        loss = loss_function(tag_scores, targets)
        # 后向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        print(index, loss.item())

        index += 1
        if index % 200 == 0:
            torch.save(m.state_dict(), './models/'+str(model_id)+'.pkl')
            model_id += 1
