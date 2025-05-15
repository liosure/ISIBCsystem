import numpy as np
from scipy import linalg
from typing import Dict, Tuple
from scipy.special import softmax


def a_func(theta: np.ndarray, K: int) -> np.ndarray:
    """等效MATLAB的a函数"""
    return np.exp(1j * 2 * np.pi * theta.reshape(-1, 1) * np.arange(K).reshape(1, -1))

def qammod(data, order: int = 16) -> np.ndarray:
    if not (np.log2(order) % 1 == 0):
        raise ValueError("M 必须是 2 的幂")
    
    # 计算星座图的维度
    k = int(np.sqrt(order))
    if k**2 != order:
        raise ValueError("M 必须是完全平方数")
    
    # 构建星座图
    real_part = 2 * (np.arange(k) - (k - 1) / 2)
    imag_part = 2 * (np.arange(k) - (k - 1) / 2)
    constellation = real_part[:, None] + 1j * imag_part[None, :]
    constellation = constellation.flatten()
    normalizeCoef = np.sqrt(np.mean(np.abs(constellation)**2))
    constellation /= normalizeCoef  # 归一化
    # 映射输入数据到星座图
    if np.any(data >= order) or np.any(data < 0):
        raise ValueError("输入数据必须在 0 到 M-1 之间")
    return constellation[data]

class TargetIdentifier:
    def __init__(self, K: int = 16, M: int = 100, N: int = 10, snr: float = 50, pilot_len: int = 10):

        self.K = K  # 阵元数
        self.M = M  # 快拍数
        self.N = N  # 子阵列数
        self.pilot_len = pilot_len # 导频长度
        self.snr = snr
        
    def generate_signal(self, theta: np.ndarray, theta_env: np.ndarray, order: int = 16, mode: str='sim') -> np.ndarray:
        """生成仿真雷达信号（等效MATLAB的数据生成部分）"""
        numSource = len(theta)
        num_env = len(theta_env)
        A = a_func(theta, self.K).T @ np.diag(np.random.randn(numSource,1).flatten())    # 增加转置操作确保矩阵维度匹配
        A_env = a_func(theta_env, self.K).T @ np.diag(np.random.randn(num_env,1).flatten()) # 增加转置操作确保矩阵维度匹配
        if mode == 'test':
            np.random.seed(42)  # 固定随机种子
        S = qammod(np.random.randint(0, order-1, (self.M * self.N,1))) ## MN X 1
        S_env = np.ones((num_env,1)) @ S.T
        S = np.ones((numSource,1)) @ S.T# numSource X MN
        pilotMat = (np.random.randn(self.N,1) + 1j*np.random.randn(self.N,1))*np.sqrt(0.5) # N X 1
        pilotMat = np.diag(pilotMat.flatten()) #N X N
        selectVector = np.zeros((self.N-1, numSource))+1
        selectIdx = np.random.choice(self.N, numSource, replace=False)
        self.sel_vec = selectIdx
        selectIdx.sort()
        for i in range(numSource):
            selectVector[selectIdx[i], i] = 1
        baseMat = np.kron((np.ones((numSource, self.N)) + selectVector.T@ pilotMat), np.ones((1, self.M))) # numSource X NM
        S = baseMat * S
        noise = (np.random.randn(self.K, self.N*self.M) + 1j*np.random.randn(self.K, self.N*self.M)) * 10**(-self.snr/20) 
        self.real_BD_theta_dict = {}
        self.esti_BD_theta_dict = {}
        for i in range(self.N):
            self.real_BD_theta_dict[f'BD{i+1}'] = []
            self.esti_BD_theta_dict[f'BD{i+1}'] = []
        for i in range(len(theta)):
            self.real_BD_theta_dict[f'BD{selectIdx[i]+1}'].append(theta[i])
        return A @ S + A_env @ S_env + noise, self.real_BD_theta_dict
        
    def _calc_mdl(self, s: np.ndarray) -> np.ndarray:
        """实现MDL准则计算"""
        n = len(s)
        mdl = np.zeros(n)
        for k in range(n):
            sum_lambda = np.mean(s[k:])
            sum_log = np.sum(np.log(s[k:]))
            mdl[k] = -self.K * sum_log + self.K * (self.N-k) * np.log(sum_lambda + np.finfo(float).eps) + 0.5*k*(2*self.N - k)*np.log(self.K)
        return mdl


    def _get_max_eig_vec(self, data: np.ndarray, threshold: float=1e-10) -> np.ndarray:
        """实现MATLAB的getMaxEigVec函数"""
        cov_matrix = data @ data.conj().T
        eigVec = np.ones((self.K, ), dtype=complex)
        residuel = 1
        while residuel > threshold:
            temp = eigVec.copy()
            eigVec = cov_matrix @ eigVec
            eigVec /= np.linalg.norm(eigVec)
            residuel = np.linalg.norm(temp - eigVec)**2
        return eigVec  # 返回最大特征值对应的特征向量

    def estimate_sources(self, data: np.ndarray) -> Dict:
        """执行信源估计"""
        u = np.zeros((self.K, self.N), dtype=complex)
        for i in range(self.N):
            input_mat = data[:, i*self.M:(i+1)*self.M]
            u[:, i] = self._get_max_eig_vec(input_mat)
        # 统一SVD参数设置
        antSpace, s, _ = linalg.svd(u)  # 确保vh维度为(K, N)
        mdl = self._calc_mdl(s**2)
        numSourceEst = np.argmin(mdl)  # 修正信源数索引偏移问题
        
        # 前向/后向子空间处理 (修正矩阵维度匹配)
        # 前向选择矩阵 (7x8)
        forward_selector = np.hstack([np.eye(self.K-1), np.zeros((self.K-1, 1))])
        # 后向选择矩阵 (7x8)
        backward_selector = np.hstack([np.zeros((self.K-1, 1)), np.eye(self.K-1)])
        
        # 构建子空间矩阵 (MATLAB风格矩阵操作)
        U_forward = forward_selector @ antSpace[:,:numSourceEst]
        U_backward = backward_selector @ antSpace[:,:numSourceEst]
        
        # 修正矩阵顺序并确保方阵
        # 确保输入矩阵为方阵
        # 最终解决方案：通过矩阵乘法确保方阵输入
        # 最终解决方案：转置矩阵确保正确维度
        solution = linalg.lstsq(U_forward, U_backward, check_finite=False)[0]
        eigvals = linalg.eigvals(solution)  # (num_source,num_source)
        theta_est = np.angle(eigvals) / (2 * np.pi)
        theta_est.sort()
        # 构建角度估计字典
        eigvec_est = a_func(-theta_est, self.K)
        omi_mat = linalg.lstsq(eigvec_est @ eigvec_est.conj().T, eigvec_est @ u)[0]
        omi_mat_fir = omi_mat / omi_mat[:, [0]]
        # 最终维度对齐的映射估计
        ma_prob_n, ma_prob_ab = self._map_estimator(omi_mat_fir)
        # 强制统一矩阵列数
        max_prob_row = self._row_map_est(ma_prob_n, ma_prob_ab)
        idx_identification = np.argmax(max_prob_row, axis=0)  # 确保索引计算正确
        # 最终正确的索引计算
        # 最终安全索引计算        
        # 构建结果容器
        for i in range(numSourceEst):
            bd_key = f'BD{idx_identification[i]}'
            angle_deg = theta_est[i]
            self.esti_BD_theta_dict[bd_key] = angle_deg
            
        return {'num_source': numSourceEst}, self.esti_BD_theta_dict, self.real_BD_theta_dict
    
    def _generate_train_data(self, theta: np.ndarray, theta_env: np.ndarray, data: np.ndarray) -> Dict:
        """生成标签字典"""
        u = np.zeros((self.K, self.N), dtype=complex)
        for i in range(self.N):
            input_mat = data[:, i*self.M:(i+1)*self.M]
            u[:, i] = self._get_max_eig_vec(input_mat)
        theta_all = np.concatenate(theta,theta_env)
        prob = np.zeros((self.N, len(theta_all)), dtype=float)
        for i in range(len(theta)):
            prob[i,self.sel_vec[i]] = 1
        prob[i,0] = 1
        theta_tag = np.concatenate((theta_all, prob), axis=0)
        return u, theta_tag

    def _map_estimator(self, omi_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """实现MATLAB的mapEstimatior函数"""
        # 最终维度安全对齐方案
        num_source = len(omi_mat)
        ma_prob_ab = np.zeros((self.N, num_source), dtype=float)
        mean_normal = np.mean(omi_mat, axis=0)
        variance_normal = np.var(omi_mat, axis=0,ddof=1)
        ma_prob_n = np.log(self.N-num_source+1)-np.log(self.N)-num_source*np.log(np.pi*variance_normal+np.finfo(float).eps)-np.sum(np.abs(omi_mat-mean_normal)**2,axis=0)/(variance_normal+np.finfo(float).eps)
        for i in range(1, num_source):
            idx = list(range(num_source))
            del idx[i]
            mean_abnormal = np.mean(omi_mat[idx,:], axis=0)
            covariance_abnormal = np.var(omi_mat[idx,:], axis=0,ddof=1)
            ma_prob_ab[:,i] = np.log(num_source-1)-np.log(self.N)-num_source*np.log(np.pi*covariance_abnormal+np.finfo(float).eps)-1/(num_source)*np.sum(np.abs(omi_mat[idx,:]-mean_abnormal)**2,axis=0)/(covariance_abnormal+np.finfo(float).eps)
        return ma_prob_n, ma_prob_ab

        
    def _row_map_est(self, ma_prob_n: np.ndarray, ma_prob_ab) -> np.ndarray:
        """实现MATLAB的rowMapEst函数"""
        # 确保输入矩阵维度一致
        prob_mat = np.concatenate((ma_prob_n.reshape(self.N,1), ma_prob_ab), axis=1)
        for i in range(self.N):
            prob_mat[i,:] = softmax(prob_mat[i,:])
        prob_mat_diff = np.ones((self.N, len(prob_mat.T)))-prob_mat
        temp = np.sum(np.log(prob_mat_diff+np.finfo(float).eps), axis=0)
        ma_prob_id_temp = temp.reshape(1,-1)-np.log(prob_mat_diff+np.finfo(float).eps)+np.log(prob_mat+np.finfo(float).eps)
        ma_prob_id = np.concatenate((temp[1:len(prob_mat.T)].reshape(1,-1), ma_prob_id_temp[:,1:len(prob_mat.T)]), axis=0)
        return ma_prob_id 

    # 保留其他方法...


##test
if __name__ == "__main__":
    t = TargetIdentifier()
    signal = t.generate_signal(np.array([0.1, 0.2, 0.3]))
    estimation_result = t.estimate_sources(signal)
    print(len(signal),len(signal.T))  # 打印估计结果
