import numpy as np
import time
import os
import pandas as pd

#声明：其中的数据生成函数和文件保存函数由AI生成

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

        # 初始化权重和偏置（使用小的随机值）
        np.random.seed(2)  # 确保可重复性

        #*0.5是为了让参数偏小（使用sm激活函数参数越接近0梯度越大）
        self.W1 = np.random.randn(self.n_x, self.n_h) * 0.5
        self.b1 = np.zeros((1, n_h))
        self.W2 = np.random.randn(n_h, n_y) * 0.5
        self.b2 = np.zeros((1, n_y))
    #激活函数
    """
    1.将结果映射于0-1区间
    2.可以让算法脱离线性叠加关系拟合复杂的函数
    """
    def sigmoid(self, z):
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    #激活函数的导数
    def Dsigmoid(self, z):
        return z * (1 - z)

    def train(self, X, y, epochs, learning_rate):
        m = X.shape[0]  # 样本数量

        for i in   在 range(epochs):
            # 前向传播
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            # 计算损失（均方误差）
            loss = np.mean((y - A2) ** 2)

            # 反向传播求得各个参数的梯度（用于后续参数更新）
            dZ2 = (A2 - y) * self.Dsigmoid(A2)
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.Dsigmoid(A1)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m

            # 更新参数
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            # 每500轮打印一次损失
            if i % 500 == 0:
                print(f"次数：{i}, 损失: {loss}")
    #根据训练参数来预测结果
    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        return A2


# ----------------------------
# 数据生成函数
# ----------------------------
def generate_large_data(num_samples=100000, input_size=5, output_size=1):
    """
    生成大规模训练和测试数据
    :param num_samples: 样本数量
    :param input_size: 输入特征数量
    :param output_size: 输出特征数量
    :return: (X_train, y_train, X_test, y_test)
    """
    np.random.seed(42)
    X = np.random.randn(num_samples, input_size) * 2  # 扩大数据范围

    # 创建更复杂的非线性关系：sin(sum(x)) + 噪声
    y = np.sin(X.sum(axis=1, keepdims=True)) + 0.1 * np.random.randn(num_samples, 1)

    # 分割训练集和测试集 (80% / 20%)
    split_idx = int(0.8 * num_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test


# ----------------------------
# 保存预测结果到 CSV
# ----------------------------
def save_results(y_true, y_pred, filename):
    """保存真实值、预测值和误差到 CSV 文件"""
    results = pd.DataFrame({
        'Actual': y_true.flatten(),
        'Predicted': y_pred.flatten(),
        'Error': (y_true - y_pred).flatten()
    })
    results.to_csv(filename, index=False)
    print(f"预测结果已保存至: {filename}")


# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == "__main__":
    # 创建结果目录
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    # 生成大数据
    print("生成数据...")
    X_train, y_train, X_test, y_test = generate_large_data(num_samples=100000, input_size=5, output_size=1)
    print(f"数据生成完成")
    print(f"train: {X_train.shape[0]} 样本")
    print(f"test: {X_test.shape[0]} 样本")

    # 创建神经网络对象(5 -> 8 -> 1)
    print("\n创建神经网络...")
    nn = NeuralNetwork(n_x=5, n_h=8, n_y=1)

    # 训练网络
    print("\n开始训练...")
    start_time = time.time()
    nn.train(X_train, y_train, epochs=2000, learning_rate=0.05)
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")

    # 测试网络
    print("\n正在测试...")
    predictions = nn.predict(X_test)
    test_error = np.mean(np.square(y_test - predictions))
    print(f"测试完成")
    print(f"测试 MSE: {test_error:.6f}")

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"nn_results_{timestamp}.csv")

    save_results(y_test, predictions, result_file)
