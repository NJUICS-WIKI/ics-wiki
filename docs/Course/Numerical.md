# NumericalAnalysis

> 随便整一篇来测试显示效果，，，

Three Easy Pieces: **插值与拟合**, **特征值与随机游走**, **最优化与线性规划**

## 方程求根
1. 二分法 猫都会做
2. 不动点迭代: 需要目标点导数比较小
3. 根的敏感性与误差分析

## 插值
---
1. 插值三问:
   - 插值是为什么? ~~好看(即答)~~ 连续性
   - 用什么插值? Polynomial
2. 经典之拉格朗日插值. 
   没啥好说的: 可证明唯一性; 误差分析 
   应用: 分享秘密: 6个人合体:还原出目标多项式; 5个人=0个人 
   龙格现象: 边界的剧烈抖动
3. 切比雪夫插值多项式: 解决龙格现象
   - 动机: 提高对插值误差的最大值的控制: $\frac{\prod_{i=1}^n(x-x_i)}{n!}f^{(n)}(c)$
   - 假设 $f: [-1, 1] \rightarrow \mathbb{R}$ \
   - 目标: 选取 $x_1, x_2, \cdots, x_n$ 最小化 $|| \prod(x-x_i) ||_{\infty}$
   - $x_i = \cos{\frac{(2i-1)\pi}{2n} }$
   - Chebyshev插值多项式 $T_n(x):=2^{n-1}(x-x_1)(x-x_2)\cdots(x-x_n)$
   - 等价于 $T_n(x):= \cos{(n \arccos x)}$
   - 递归关系: $T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)$
   - 可以证明的是, 这是令误差最小的插值方式

## 逼近
---
1. Weierstrass定理: 连续函数可以任意地被多项式逼近
2. Bernstein Polynomial:
   - 给定: $f: [0,1] \rightarrow R$
   - 目标: 构造多项式 $p$, 使得在 $x \in [0,1]$ 这一点上, $p(x)$ 尽可能接近 $f(x)$.
   - $K \sim B(n, x)$
   - $E[k]=nx$, $Var[K]=nx(1-x), Pr[K=i] = {n\choose i} x^i(1-x)^{n-i}$
   - Bernstein Poly的一组基: $b_{n,i}= {n\choose i} x^i(1-x)^{n-i}$
   - $\displaystyle \lim_{n\rightarrow \infty}Pr[|\frac K n - x| > \delta]$ 对所有 $x$ 一致成立 
   - Bernstein Poly的本体: $E[f(\frac K n)] = \sum_{i=0} f(\frac i n)b_{n, i}(x)$ 
     - 个人理解: 等距采样, 加权平均(权是一个分布多项式函数)
   - 证明: 拆成两部分

## 范数
---
1. 一般向量范数: $\displaystyle ||x||_k = (\sum{|x_i|^k})^{\frac1 k}$
2. 函数范数: $||f||_{\infty} = \sup |f|$, $||f||_2 = (\int |f(x)|^k dx)^{\frac 1 k}$ 

## 最小二乘法
---
1. 几何: 简单投影 $A^TA\overline{x} = A^Tb$
2. 微积分: 最小化 $(Ax-b)^T(Ax-b)$, 求解梯度 $\Delta = 2A^TAx-2A^Tb = 0$.
3. Gram-Schmidt: 消除已经存在的维度上分量. 

## 内积空间，正交性
---
1. 条件: 对称 线性 正定
2. 正交: $<\phi_i, \phi_j> = \begin{cases} C_i & i = j \\ 0 & i \neq j \end{cases}$
3. Chebyshev多项式关于 $w(x) = \frac{1}{\sqrt{1-x^2}}$ 所定义的内积正交: 
   ${ < f , g > }_w = \int_{-1}^1 f(x) g(x) \frac{dx}{\sqrt{1-x^2}}$
4. 三角级数正交: 

## 狐里叶变换
---
1. DFT FFT IFFT
2. 动机: 多项式乘法: $p(x) * q(x)$ 
   1. 给定 $p(x), q(x)$ 的系数
   2. 令 $m \geqslant 2n+1$, 选定系数 $x_0, x_1, \cdots, x_m$, 计算 $p,q $ 在上面的点值. 
   3. 计算 $r(x_i) = p(x_i)q(x_i)$
   4. 对 $r$ 插值, 还原出 $r$ 的系数
3. 愿景: 2, 4 都能做到 $O(n\log n)$
4. DFT: 
   1. 选取 $x_i$ 为 $m+1$ 次单位根. 
   2. 逆变换: 插值
   3. 但是, 还是很慢,,,
5. FFT: 
   1. 引入分治, 将 $p(x)$ 分成奇数和偶数:
      - $E(z) = a_0 + a_2z + \cdots$
      - $O(z) = a_1 + a_3z + \cdots$
      - $p(z) = E(z^2) + zO(z^2)$
   2. IFFT:
      - DFT, FFT: $p(\omega^l)= \frac1{\sqrt{n+1}}\sum a_j \omega_{lj}$
      - IFFT: $a_l = \sum p(\omega^j) \omega^{-lj}$, 和DFT唯一区别是 $\omega \rightarrow \omega^{-1}$

## 解方程组之迭代Method
1. 线性方程组的数值稳定性: $Ax = b$
   - 定义条件数 $cond(A) = \frac{x相对误差}{b相对误差}$
   - $cond(A) = \max{\frac{||A^{-1}e||/||A^{-1}b||}{||e|| / ||b||}} = \max{\frac{||A^{-1}e||}{||e||}} \max{\frac{||Ax||}{||x||}}$
   - 引入矩阵的算子范数 $||A||:= \max{\frac{||Ax||}{||x||}}$
      - 特别地有: $||A||_2:= \max{\frac{||Ax||_2}{||x||_2}} = \sqrt{\lambda_{max}(A^TA)}$
   - $cond(A) = ||A|| \cdot ||A^{-1}||$
2. 高斯消元法: 朴素的 $O(n^3)$ 做法
3. Jacobi迭代:  
   - 收敛的充分条件: 严格对角占优(对角线上元素超过行内其他元素)
   - 不动点迭代形式: $A=L+D+U$, $Ax=b$ 可以写为 $x=D^{-1}(b-(L+U)x)$
   - Gauss-Seidel: $(L+D)x=b-Ux$
   - 收敛性: $x_{k+1} = Ax_{k} + b$ 
     - 令 $x_*$ 为一个不动点
     - 有 $x_k-x_{ * } = A^k(x_0-x_{ * })$
     - 收敛条件:
       - $A^k \rightarrow 0$
       - $x_0 - x_*$ 存在于一个好的线性子空间里.   
     - 谱半径: 最大的{特征值的绝对值}
   - Jacobi迭代的谱半径: $\rho(D^{-1}(L+U))$, 若满足严格对角占有, 可以证明$\rho < 1$
   - Moore-Penrose伪逆 $\overline x = (A^TA)^{-1}A^Tb$, $(A^TA)^{-1}A^T$ 是维尼.(前提: 列线性无关)
4. Richardson迭代:
   - 目的: $A^{-1} = p(A)$ 的近似.
   - 等价地, $q(x) = 1-xp(x)$, 需要寻找 $q(0) = 1$, $q(x) \approx 0, \forall x > 0$
   - 要解 $Ax=b$, 即找出 $x_* = p(A)b \in span(b, Ab, A^2b, \cdots)$
   - 迭代形式: $x_{k+1} = (I -\alpha A)x_k + \alpha b$
   - 误差 $e_k = x_k - x_*$ 满足 $e^k = (I-\alpha A)e_{k-1} = (I-\alpha A)^k e_0$
   - 收敛性: 当且仅当 $\rho(I-\alpha A) = |1-\alpha\lambda_{max}| < 1$ 
   - 这里的 $x_k$ 处于 $span(b, Ab, A^2b, \cdots, A^{k-1}b)$ 中(Krylov子空间)
   - Chebyshev迭代: 每一轮选取不同的 $\alpha$, 最小化 $||\prod_i (I-\alpha_i A)||$, 使用了Chebyshev多项式. 
   - 内积: 给定正定矩阵 $A$, 定义内积 $ { < x, y > }_A := x^TAy $.
     - 内积为0, 则称 x,y 关于 A 共轭.
     - 共轭向量一定线性无关.
   - 考虑不断增大的Krylov子空间序列, 设 $x_i$ 为第 i 个子空间中最逼近 $x_*$ 的元素.
   - 引理 $v_i := x_i - x_{i-1}$, 则  \{v_i\} 两两共轭. (注意到$Ax_j - b$ 必然于 $K_j$ 正交, 因此 $Av_j = A(x_j - x_{j-1}) \in K_{j-1}^{不会打的符号}$, 而 $v_i$ \in K_i \in K_{j-1}, 因此这两个向量共轭) 
   - 推论 $K_i = span(v_1, v_2, \cdots, v_i)$
5. 幂迭代(计算特征值)
   - 考虑任意向量的特征值分解, 可以找出最大特征值和对应的占优向量,
   - 迭代可能造成浮点数溢出, 一般每次迭代选择进行归一化.
   - 迭代得到了一个近似的特征向量方向, 求解最小二乘法 $||Ax-\lambda x||_2^2$, 法线方程为 $\lambda = \frac{x^TAx}{x^Tx}$, Rayleigh Quotient.
   - 定理, 如果上述目标函数误差小于 $\epsilon$, 则特征值求解误差同样小于 $\epsilon$. 

## 矩阵与特征值
1. 正定/半正定 $\Leftrightarrow$ 特征值正/非负
2. 特征值的min-max刻画: 最大的特征值满足: $\lambda_n(A) = \max \frac{x^TAx}{x^Tx}$(换成min也可以, 只需要考虑谱分解)
3. 图相关矩阵: 
   1. A: 邻接矩阵, D: 度数矩阵, L = D-A: Laplacian
4. $cond_2(A) = ||A||_2 ||A^{-1}||_2 = \lambda_{max} / \lambda_{min}$
5. Cayley-Hamilton:
   - $Ap(A) = I$
   - $q(x) = 1 - xP(x)$, $q(A) = 0$
   - 如何近似 $q$? Richardson Iteration
6. 奇异值分解SVD (这里好像换了个名字)

## 随机游走 Random Walk
1. 定义: 从一个点出发, 每一次都随机走到一个邻居上, 不断重复.
2. 长期表现会怎样呢?
   1. 重复t步后, 当前顶点是某个u的概率是多少?
   2. 是否存在稳态分布?
   3. 多久才会收敛? (mixing time)
   4. 从s出发多久才到t? (hitting time)
   5. 多久才能遍历每个节点各一次? (cover time)
   1,2,3都可以通过纯概率的方法去解决, 4,5见电阻网络.
3. 马尔可夫链: ~~past is in the past~~ 一切之和当前状态有关.
   1. 转移矩阵描述: $\vec{p}_{t+1} = P\cdot \vec{p}_{t}$
   2. 对于有限Markov Chain, 若对应的有向图是强连通的, 那说明它是不可约的.
   3. 如果Markov Chain中每个状态都是非周期的, 那么这个Chain是非周期的.
   4. 引理: 对于不可约, 非周期Markov Chain, 一定存在足够大的常数, $P^t_{i,j} > 0, \forall t \geqslant T$
   5. 稳态: $\vec{\pi} = \vec{\pi}\cdot P$.
   6. 回归时间: 从 $i$ 出发, 第一次返回到 $i$ 的时间 $H_i$. $h_i = E[H_i]$
   7. Markov基本定理: 有限, 非周期, 不可约的Markov链, 必然收敛到一个唯一的稳态分布, $\pi(i) = \frac 1 h_i$. (怎么证,,,,)
   8. 例子: PageRank
   9. Markov相关内容详细证明见[本书](https://box.nju.edu.cn/d/62bf772ba48d4a598311/files/?p=%2F%E8%AE%A1%E7%AE%97%E6%96%B9%E6%B3%95%2FFinite%20Markov%20Chains%20and%20Algorithmic%20Applications.pdf)
4. 无向图上的随机游走
   1. $p_{t+1}(v) = \sum_{u:uv\in E}p_t(u)\cdot \frac1{deg(u)} $
   2. $p_{t+1} = (AD^{-1}p_t)$, $p_t = (AD^{-1})^t p_0$
   3. 稳态分布: $\vec{\pi} = \frac{\vec{d}}{2m}$, 是 $AD^{-1}$ 的一个右特征向量, 特征值为1.
   4. 无向图上的马尔可夫基本定理: 对于任意有限的, 连通的, 非二分图, 不管从什么养的 $p_0$ 开始, 总有 $\vec{p_t} \rightarrow \vec{\pi}$.
   5. 惰性随机游走: $p_t = \frac12(I + AD^{-1})^t p_0$ (对二分图有效)
   6. $W$ 与 $D^{-\frac12}WD^{\frac12} = D^{-\frac12}AD^{-\frac12} = \mathcal{A}$ 相似, 拥有相同的特征值.、
      1. 对于d正则图比较特殊, $W = \frac 1 d A= I- \frac 1 d L$, 可以将 $p_0$ 展开, 从大到小排序的特征值$\alpha_1 = 1$, 特征值位于 $[-1, 1]$ 中, $a_n = -1$ 当且仅当图是二分图.
      2. 对联通的非二分图, $p_t \rightarrow c_1v_1 = \frac 1 n \vec{1} = \vec{\pi}$
      3. 对惰性的随机游走, 特征值发生了 $[-1,1] \rightarrow [0,1]$ 的保持相对距离的映射.
   7. 混合时间:
      - 定义谱间隔 $\lambda = \min{1-\alpha_2, 1-|\alpha_n|}$
      - 定理: $\epsilon-$ 混合时间.  

## 图与代数
1. 二分图邻接矩阵: $A$ 特征值关于 $x=0$ 对称.
2. 图邻接矩阵最大特征值 $deg_{avg}(G) \leqslant \lambda_{max}\leqslant deg_{max}(G)$
3. Laplacian Matrix 的一个特征值是 $0$, 对应特征向量是 $\vec{1}$, 是半正定的.
4. 当且仅当特征值 $0$ 的重数为1时, $G$ 连通. $x^TLx = \sum(x_i-x_j)^2$
5. Perron-Frobenius定理: 对于非负, 不可约, 非周期的矩阵 
   1. 最大特征值重数为1
   2. 对应特征向量中, 每个维度都是非0且同号的.
   3. $|\lambda_i| < \lambda_1$

## 电阻与电路网络
1. ~~为什么我会在这复习物理QAQ~~ 解决之前的留下的 Hitting&Cover Time 问题.
2. 基本概念
   1. 基尔霍夫定律: in=out
   2. 欧姆定律: $\Delta \phi = ir$ 
   3. 电阻 $r_e$, 电导率 $w_e = \frac1{r_e}$
   4. $b_v$ 表示外界流到 $v$ 的电流
3. 若 $w_{uv} = 1$(电阻为1), 求解$\vec{b} = L \vec{\phi}$ (That's All We Want)
4. $L = BWB^T$, $b= L\phi = BWB^T\phi = B\vec{i}$
5. 若存在满足条件的解, 则 $b = L\phi \perp \vec{1}$ (外部流出和流入的相等) (其实这是个充要条件)
6. $L^{+} = \sum_{i=2}^n \frac1{\lambda_i}v_iv_i^T$ 是维尼. 全体解集是 $\{L^+\vec{b} + c \vec{1} \}$, 若固定一点电势, 有唯一解.
7. $R_{eff}$ 等效电阻为电势差. $R_{eff}(s,t) = b^T_{st}L^+b_{st}$, $b_{st}(s) = 1, (t=-1)$
8. Thompson's Principle: $R_{eff}(s,t) \leqslant \epsilon(\vec{g})$, 单位电流最小化能量.
9. Commute Time: $C_{s,t} = 2mR_{eff}(s,t)$, $m = |E(G)|$
   1.  推论: $C_{u,v} \leqslant 2m$
   2.  连通图遍历时间最多为 $2m(n-1)$
- 有种感觉越往下写越详细了,,, 虽然并不是我的本意, 所以————

## 线性规划+凸优化
上学期凸优化覆盖了这些了(PS:真的吗), 摸了

## 课程评价
1. 难度: 魔改的比较猛烈, 所以还是有一定难度的
2. 有趣程度: 除了线性规划+凸优化之外, 其他都算比较新的内容, 我觉得随机游走部分比较有意思
3. 其他: 可能是第一年开课, 作业和讲课内容还有些脱节, 完成作业比较考验数理基础
4. 大概就是这么多了, 今晚研究下习题, 明天就进入算法了(四天分给除了数理逻辑以外四门www)