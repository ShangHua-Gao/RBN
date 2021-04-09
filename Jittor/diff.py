import torch
import jittor as jt
import numpy as np
from resnet_jittor import resnet101 as r101_jittor
from resnet_torch import resnet101 as r101_torch
import time

jt.flags.use_cuda = 1

# 定义numpy输入矩阵
bs = 1
test_img = np.random.random((bs,3,224,224)).astype('float32')

# 定义 pytorch & jittor 输入矩阵
pytorch_test_img = torch.Tensor(test_img).cuda()
jittor_test_img = jt.array(test_img)

# 跑turns次前向求平均值
turns = 100

# 定义 pytorch & jittor 的xxx模型，如vgg
pytorch_model = r101_torch().cuda()
jittor_model = r101_jittor()

# 把模型都设置为eval来防止dropout层对输出结果的随机影响
pytorch_model.eval()
jittor_model.eval()

# jittor加载pytorch的初始化参数来保证参数完全相同
# jittor_model.load('resnet101_rbn_best_model.pth')
# pytorch_model.load_state_dict(torch.load('resnet101_rbn_best_model.pth'))

# 测试Pytorch一次前向传播的平均用时
for i in range(10):
    pytorch_result = pytorch_model(pytorch_test_img) # Pytorch热身
torch.cuda.synchronize()
sta = time.time()
for i in range(turns):
    pytorch_result = pytorch_model(pytorch_test_img)
torch.cuda.synchronize() # 只有运行了torch.cuda.synchronize()才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
end = time.time()
tc_time = round((end - sta) / turns, 5) # 执行turns次的平均时间，输出时保留5位小数
tc_fps = round(bs * turns / (end - sta),0) # 计算FPS
print(f"- Pytorch forward average time cost: {tc_time}, Batch Size: {bs}, FPS: {tc_fps}")

# 测试Jittor一次前向传播的平均用时
for i in range(10):
    jittor_result = jittor_model(jittor_test_img) # Jittor热身
    jittor_result.sync()
jt.sync_all(True)
# sync_all(true)是把计算图发射到计算设备上，并且同步。只有运行了jt.sync_all(True)才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
sta = time.time()
for i in range(turns):
    jittor_result = jittor_model(jittor_test_img)
    jittor_result.sync() # sync是把计算图发送到计算设备上
jt.sync_all(True)
end = time.time()
jt_time = round((time.time() - sta) / turns, 5) # 执行turns次的平均时间，输出时保留5位小数
jt_fps = round(bs * turns / (end - sta),0) # 计算FPS
print(f"- Jittor forward average time cost: {jt_time}, Batch Size: {bs}, FPS: {jt_fps}")

threshold = 1e-3
# 计算 pytorch & jittor 前向结果相对误差. 如果误差小于threshold，则测试通过.
x = pytorch_result.detach().cpu().numpy() + 1
y = jittor_result.data + 1
relative_error = abs(x - y) / abs(y)
diff = relative_error.mean()
assert diff < threshold, f"[*] forward fails..., Relative Error: {diff}"
print(f"[*] forword passes with Relative Error {diff}")