1. LR线性回归的原理和推导
2. XGBoost原理及其推导
  是一种基于梯度提升决策树（Gradient Boosting Decision Trees, GBDT）的高效实现。GBDT是一种集成学习方法，它通过逐步构建多个决策树，每棵树都是在前一棵树的基础上进行改进。具体来说，GBDT使用梯度下降的思想来最小化损失函数，逐步调整模型的预测值。

GBDT的基本公式为：![image](https://github.com/user-attachments/assets/4eb759a1-5172-45f8-8352-8f848a087bd2) 每一个f(x)目标是通过每棵树的学习来减少上一棵树的误差
