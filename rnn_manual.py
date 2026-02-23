import numpy as np


# short term memory
class ST_Memory:
    def __init__(self):
        pass
    def add_memory(self):
        '''持续添加短期记忆'''
        '''人一直在记忆和遗忘，就算只是一瞥，仔细想想也会回忆起来'''
        pass
    def get_st_memory(self, x):
        '''根据输入取出合适的短期记忆'''
        pass


# ?如何从 词语的权重 推广到 知识点的
class AttentionMemory:
    def __init__(self, input_size, hidden_size, output_size, num_blocks=3):
        # 使用多块参数, 通过注意力加权组合
        self.num_blocks = num_blocks
        
        # 为每个记忆块创建一个key（查询向量）
        self.keys = np.random.randn(num_blocks, input_size) * 0.1
        
        # 初始化多个记忆块
        self.blocks = []
        for _ in range(num_blocks):
            w_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
            w_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
            w_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
            b_h = np.zeros((hidden_size, 1))
            b_y = np.zeros((output_size, 1))
            self.blocks.append([w_xh, w_hh, w_hy, b_h, b_y])
    
    def get_memory(self, x, temperature=1.0):
        """
        使用注意力机制选择记忆
        x: 当前输入向量 (input_size, 1)
        temperature: softmax温度参数，控制选择集中度
        """
        # 将x展平为(input_size,)
        x_flat = x.flatten()
        
        # 计算输入与每个key的相似度（点积）
        similarities = []
        for i in range(self.num_blocks):
            similarity = np.dot(x_flat, self.keys[i])  # 计算点积
            similarities.append(similarity)
        
        # 应用softmax得到权重
        similarities = np.array(similarities) / temperature
        exp_similarities = np.exp(similarities - np.max(similarities))  # 数值稳定性
        weights = exp_similarities / np.sum(exp_similarities)
        
        # 加权平均参数
        weighted_params = None
        for i in range(self.num_blocks):
            params_i = self.blocks[i]
            weight_i = weights[i]
            
            if weighted_params is None:
                weighted_params = [p * weight_i for p in params_i]
            else:
                for j in range(len(params_i)):
                    weighted_params[j] += params_i[j] * weight_i
        
        return weighted_params, weights
    
    def get_from_disk(self):
        '''从磁盘中取得数据'''
        pass

    def update_memory(self, block_idx, grads, learning_rate):
        """
        按 block_idx 对应的参数块进行梯度更新。
        grads: [dW_xh, dW_hh, dW_hy, db_h, db_y]
        """
        W_xh, W_hh, W_hy, b_h, b_y = self.blocks[block_idx]
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads

        self.blocks[block_idx][0] = W_xh - learning_rate * dW_xh
        self.blocks[block_idx][1] = W_hh - learning_rate * dW_hh
        self.blocks[block_idx][2] = W_hy - learning_rate * dW_hy
        self.blocks[block_idx][3] = b_h  - learning_rate * db_h
        self.blocks[block_idx][4] = b_y  - learning_rate * db_y


class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, memory : AttentionMemory):
        # 初始化参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.memory = memory
        # 运行时缓存
        self.x_sequence = None
        self.h_states = None
        self.outputs = None
        self.params_per_t = None
        self.attn_weights = None
        
    def forward(self, x_sequence):
        """
        前向传播
        x_sequence: 形状为 (seq_len, input_size, 1) 的列表
        返回: 输出序列和隐藏状态序列
        """
        self.x_sequence = x_sequence
        seq_len = len(x_sequence)
        h_prev = np.zeros((self.hidden_size, 1))  # 初始隐藏状态
        
        self.h_states = [h_prev]  # 存储所有时间步的隐藏状态
        self.outputs = []
        self.params_per_t = []
        self.attn_weights = []
        
        for t in range(seq_len):
            (W_xh, W_hh, W_hy, b_h, b_y), weights = self.memory.get_memory(x_sequence[t])
            self.params_per_t.append((W_xh, W_hh, W_hy, b_h, b_y))
            self.attn_weights.append(weights)

            # 计算当前隐藏状态
            h_t = np.tanh(
                W_xh @ x_sequence[t] + 
                W_hh @ h_prev + 
                b_h
            )
            # 计算输出
            y_t = W_hy @ h_t + b_y
            
            # 保存
            self.h_states.append(h_t)
            self.outputs.append(y_t)
            
            # 更新前一个隐藏状态
            h_prev = h_t
            
        return self.outputs, self.h_states[1:]
    
    
    def backward(self, targets, learning_rate=0.01):
        """
        通过时间反向传播
        现在梯度会更新到memory中对应的block
        """
        if self.x_sequence is None or self.h_states is None or self.outputs is None:
            raise ValueError("需要先调用forward进行前向传播")
            
        seq_len = len(self.x_sequence)
        num_blocks = self.memory.num_blocks
        
        # 初始化梯度（为每个block存储梯度）
        gradients = [
            [
                np.zeros_like(self.memory.blocks[i][0]),
                np.zeros_like(self.memory.blocks[i][1]),
                np.zeros_like(self.memory.blocks[i][2]),
                np.zeros_like(self.memory.blocks[i][3]),
                np.zeros_like(self.memory.blocks[i][4])
            ]
            for i in range(num_blocks)
        ]
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        # 反向传播
        for t in reversed(range(seq_len)):
            # 获取当前时间步使用的加权参数与注意力权重
            W_xh, W_hh, W_hy, b_h, b_y = self.params_per_t[t]
            weights = self.attn_weights[t]
            
            # 输出层的梯度
            dy = self.outputs[t] - targets[t]  # 假设是回归任务
            dW_hy = dy @ self.h_states[t+1].T
            db_y = dy
            
            # 隐藏层的梯度
            dh = W_hy.T @ dy + dh_next
            
            # 通过tanh的梯度
            dh_raw = (1 - self.h_states[t+1] ** 2) * dh
            dW_xh = dh_raw @ self.x_sequence[t].T
            dW_hh = dh_raw @ self.h_states[t].T
            db_h = dh_raw
            
            # 传递到前一时间步
            dh_next = W_hh.T @ dh_raw

            # 将梯度按注意力权重分配给各个block
            for block_idx in range(num_blocks):
                w = weights[block_idx]
                gradients[block_idx][0] += w * dW_xh
                gradients[block_idx][1] += w * dW_hh
                gradients[block_idx][2] += w * dW_hy
                gradients[block_idx][3] += w * db_h
                gradients[block_idx][4] += w * db_y
        
        # 更新memory中所有涉及的block
        for block_idx, grads in enumerate(gradients):
            # 应用梯度裁剪，防止梯度爆炸
            for grad in grads:
                np.clip(grad, -5, 5, out=grad)
            
            # 更新memory中的block
            self.memory.update_memory(block_idx, grads, learning_rate)


def _toy_run():
    """简易示例：随机数据跑一遍前向和反向，验证不会报错"""
    np.random.seed(42)
    input_size, hidden_size, output_size = 5, 4, 3
    seq_len = 6

    memory = AttentionMemory(input_size, hidden_size, output_size, num_blocks=3)
    rnn = VanillaRNN(input_size, hidden_size, output_size, memory)

    x_sequence = [np.random.randn(input_size, 1) for _ in range(seq_len)]
    targets = [np.random.randn(output_size, 1) for _ in range(seq_len)]

    outputs, _ = rnn.forward(x_sequence)
    rnn.backward(targets, learning_rate=0.05)
    # 返回最后一步输出用于简单检查
    return outputs[-1]


if __name__ == "__main__":
    last_out = _toy_run()
    print("toy run ok, last output shape:", last_out.shape)

