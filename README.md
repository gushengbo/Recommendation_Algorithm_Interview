推荐算法经典模型![image](https://github.com/user-attachments/assets/27be482c-8a3d-4c9b-ada3-33d19a803a33)

一、DIN模型
DIN模型出发点：作为阿里发布的第一个兴趣建模模型，其出发点是基于阿里妈妈团队在其业务数据中观察到的Diversity和Local activation现象。

Diversity和Local activation 名词理解：

Diversity指用户历史兴趣呈多峰分布。即用户在过去的一段时间中，会对各种行然不同的商品产生兴趣，并产生浏览、点击、加购、下单、复购等指示性消费信息。
考虑我们实际使用淘宝场景。如果我们在短期内想购买多种不同类别的商品，如水果、鞋子、笔记本电脑，并产生了大量相关的访问记录。那么在当前时刻，我们至少存在三个截然不同的兴趣峰。
Local activation则指的是，尽管用户历史兴趣呈多峰分布，但决定用户当下转化只会是其中的某几个兴趣导致。
在最终下单笔记本电脑前，我们产生了很多的水果、鞋子以及笔记本的选货记录。但在购买笔记本这件事情上，只有和笔记本相关的访问中，才可能隐藏我们可能下单的兴趣信息。

![image](https://github.com/user-attachments/assets/ef906f2c-cde7-4ace-8fb3-3f6daf69938c)
如上图1-2左是阿里2016提出的GwEN，其代表着业界对用户行为序列处理的流行做法，会将用户的历史行为直接pooling后输入MLP部分。而DIN则是通过activation unit计算出candidate与用户历史行为的兴趣度分值，并以此分值加权做sum-pooling，然后再将pooling后的结果与商品candidate的embedding concatenate后输入MLP。

1. LR线性回归的原理和推导
2. XGBoost原理及其推导
  是一种基于梯度提升决策树（Gradient Boosting Decision Trees, GBDT）的高效实现。GBDT是一种集成学习方法，它通过逐步构建多个决策树，每棵树都是在前一棵树的基础上进行改进。具体来说，GBDT使用梯度下降的思想来最小化损失函数，逐步调整模型的预测值。

GBDT的基本公式为：![image](https://github.com/user-attachments/assets/4eb759a1-5172-45f8-8352-8f848a087bd2) 每一个f(x)目标是通过每棵树的学习来减少上一棵树的误差



3. 生成式检索相比于传统检索的优势。基于大模型，可以学习到文本潜在的高级语义特征。达到更个性化的检索、推荐。
4. 生成式检索baseline是unimo-text-1.0-large
5. 每个广告都包含落地页特征文本和核心词特征文本。训练集是搜索词对应的广告id。所以我们直观想法就是，得让模型先充分学习所有广告的文本信息，然后再去训练训练集。落地页跟核心词都是一长文本，我们截取成短文本，以匹配搜索词的长度。落地页特征是比较粗糙的特征，每一句话跟广告不一定完全匹配，比如（新能源汽车的广告，落地页特征可能是，xxx汽车店，4S店。），而且会有几条非常相似的广告，对应的落地页特征完全一样。核心词特征也是类似，但会比落地页特征更准确一些。多阶段学习相当于让模型粗略地记忆所有广告的特征，然后再通过训练集去更加精细地记忆广告。
6. DPR内部结构是两个BERT模型，六层transformer的encoder（是什么？） BM25是稀疏检索模型，通过关键词匹配。
7. 损失函数，先softmax, 再交叉熵。采用余弦相似度，因为可以避免向量长度的影响。
8. TSNE是什么？
9. 使用Neural-Chat做数据增强，因为问题只有很简短的一句话，去匹配论文的摘要。用大语言模型补充更多的信息，去匹配长度相似的论文摘要。
10. 迭代伪标签，充分利用无标签数据集。

Transformer
![image](https://github.com/user-attachments/assets/a1b51ce4-92bb-4e41-a95c-8b66b8295894)

  11. transformer encoder-decoder， encoder用多头注意力机制和MLP（对每个单词做投影）为一个block，6个block合成一个encoder。
  12. decoder也是6个，有一个自回归的东西，就是预测t时刻的时候，前t-1时刻的输出也作为输入，并且有mask，就是为了mask掉t时刻以后的输入。sequence mask 是为了使得 decoder 不能看见未来的信息。什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。multi-head就是模仿cnn的多通道。用多个dot-product attention 并行计算(多个linear，多种投影)，然后concat，再输入到线性层。
  13. 自注意力机制：输入到encoder的k,v,q矩阵的embedding都是同一个，也就是句子每个单词编码成embedding。每个query跟所有key做相似度，得到的权重再乘以所有的value，得到对应那个query位置的输出。
  14. 注意：
  1、在Encoder中的Multi-Head Attention也是需要进行mask的，只不过Encoder中只需要padding mask即可，而Decoder中需要padding mask和sequence mask。
  2、Encoder中的Multi-Head Attention是基于Self-Attention地，Decoder中的第二个Multi-Head Attention就只是基于Attention，它的输入Quer来自于Masked Multi-Head Attention的输出，Keys和Values来自于Encoder中最后一层的输出。
  15. position encoding, 加入时序信息。(sin,cos,sin....)长度512，与input embedding相加

BERT
双向的，之前的语言模型都是单向预测的。MLM和NSP
t-SNE降维方法https://zhuanlan.zhihu.com/p/426068503  保证降维前后它们是否具有相同的距离结构
