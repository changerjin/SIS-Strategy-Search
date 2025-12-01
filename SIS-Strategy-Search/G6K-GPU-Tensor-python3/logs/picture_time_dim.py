import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from math import log

def default_dim4free_fun(blocksize):
    """
    Return expected number of dimensions for free, from exact-SVP experiments.

    :param blocksize: the BKZ blocksize

    """
    return max(int(blocksize / log(blocksize) - 5), 0)
    # return int(11.5 + 0.075*blocksize)

# 初始化存储数据的数组
blocksize = []
time = []

# 读取日志文件
log_file = 'Test_Pump_Average_Time_900.log'

try:
    with open(log_file, 'r') as f:
        # 遍历文件的每一行
        for line in f:
            # 使用正则表达式匹配目标格式的行
            # 匹配类似 "bkz-front-60: average time: 1.920" 的行
            match = re.search(r'bkz-front-(\d+): average time: (\d+\.\d+)', line)
            if match:
                # 提取第一个数字（blocksize）和第二个数字（time）
                bs = int(match.group(1))
                t = float(match.group(2))
                blocksize.append(bs)
                time.append(log(t, 2))
    
    # 检查是否成功提取到数据
    if not blocksize or not time:
        print("没有找到符合格式的数据")
    else:
        blocksize = [b - default_dim4free_fun(b) for b in blocksize]
        print(blocksize)
        # 转换为numpy数组以便后续处理
        blocksize_front = []
        time_front = []
        blocksize_back = []
        time_back = []
        intersection = 60
        for i, b in enumerate(blocksize):
            if b >= intersection-1:
                blocksize_back.append(b)
                time_back.append(time[i])
            if b <= intersection+1:
                blocksize_front.append(b)
                time_front.append(time[i])

        y = np.array(time)
        x = np.array(blocksize)
        
        y_back = np.array(time_back)
        x_back = np.array(blocksize_back)

        y_front = np.array(time_front)
        x_front = np.array(blocksize_front)

        # 绘制散点图
        plt.scatter(x, y, color='blue', label='Pump in BKZ')
        

        # 线性回归拟合趋势线
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        trend_line = slope * x + intercept
        
        slope_back, intercept_back, r_value_back, p_value_back, std_err_back = stats.linregress(x_back, y_back)
        trend_line_back = slope_back * x_back + intercept_back

        slope_front, intercept_front, r_value_front, p_value_front, std_err_front = stats.linregress(x_front, y_front)
        trend_line_front = slope_front * x_front + intercept_front

        # 绘制趋势线
        plt.plot(x, trend_line, color='red', label=f'trend: y = {slope:.2f}x + {intercept:.2f}')
        plt.plot(x_back, trend_line_back, color='green', label=f'trend (blocksize>={intersection}): y = {slope_back:.2f}x + {intercept_back:.2f}')
        plt.plot(x_front, trend_line_front, color='orange', label=f'trend (blocksize<{intersection}): y = {slope_front:.2f}x + {intercept_front:.2f}')


        # 添加图表标题和坐标轴标签
        plt.title('Blocksize vs Time')
        plt.xlabel('Blocksize')
        plt.ylabel('Time (sec.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图表
        plt.show()
        
        # 输出拟合结果
        print(f"线性回归方程: y = {slope:.4f}x + {intercept:.4f}")
        print(f"相关系数 r: {r_value:.4f}")
        print(f"R²: {r_value**2:.4f}")
        
except FileNotFoundError:
    print(f"错误：找不到文件 {log_file}")
except Exception as e:
    print(f"发生错误：{str(e)}")
