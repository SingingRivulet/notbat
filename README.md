# notbat
基于PSPNet的midi转录。代码不完整（只有模型、训练部分，没有测试部分）  
依赖包只有torch  
大部分为原生代码，请手动make  
未经过对比测试  
效果如何，不知道  
性能如何，不知道  
怎么使用，不知道  

调用方法  
```
    m = model.PSPNet(3).to(device=device)  # 创建模型
    weights = torch.load('./models/61.pkl', map_location=torch.device(device))# 加载模型路径
    m.load_state_dict(weights)
    输出midi = m(频谱数据)
```
  
模型下载：  
链接: https://pan.baidu.com/s/1vednuw9hiagOzXPaWmOEvA  密码: 869q  
