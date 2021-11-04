import * as tf from '@tensorflow/tfjs';

// 代码清单2-1.将数据转换成张量
const trainData = {
    sizeMB: [
        0.080, 9.000, 0.001, 0.100, 8.000,
        5.000, 0.100, 6.000, 0.050, 0.500,
        0.002, 2.000, 0.005, 10.00, 0.010,
        7.000, 6.000, 5.000, 1.000, 1.000
    ],
    timeSec: [
        0.135, 0.739, 0.067, 0.126, 0.646,
        0.435, 0.069, 0.497, 0.068, 0.116,
        0.070, 0.289, 0.076, 0.744, 0.083,
        0.560, 0.480, 0.399, 0.153, 0.149
    ]
};

const testData = {
    sizeMB: [
        5.000, 0.200, 0.001, 9.000, 0.002,
        0.020, 0.008, 4.000, 0.001, 1.000,
        0.005, 0.080, 0.800, 0.200, 0.050,
        7.000, 0.005, 0.002, 8.000, 0.008
    ],
    timeSec: [
        0.425, 0.098, 0.052, 0.686, 0.066,
        0.078, 0.070, 0.375, 0.058, 0.136,
        0.052, 0.063, 0.183, 0.087, 0.066,
        0.558, 0.066, 0.068, 0.610, 0.057
    ]
}

// 代码清单2-2.将数据转换成张量
const trainTensors = {
    sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(trainData.timeSec, [20, 1])
};

const testTensors = {
    sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(testData.timeSec, [20, 1])
};

// 2.1.4 定义简单的模型
// 模型|网络 model - 输入特征映射到输出目标上的函数
// 回归 regression - 模型会输出实数值,并且会尝试匹配训练集中的目标
// 分类 classification - 输出一系列选项中的做出的选择
//
// 代码清单2-3.构建线性回归模型
const model = tf.sequential();
model.add(tf.layers.dense({
    inputShape: [1],
    units: 1
}));

// 代码清单2-4.配置训练选项:模型编译
// 随机梯度下降算法 stochastic gradient descent
// 此处会用微积分计算结果来对模型做出相应调整,然后重复这一流程
model.compile({
    optimizer: 'sgd',
    loss: 'meanAbsoluteError'
});

// 2.1.5 使模型拟合训练集
// Tensorflow.js可以通过调用模型的fit()方法来训练模型,让模型更好地拟合训练集
// 代码清单2-5 拟合线性回归模型
// 立即调用异步函数表达式 immediately invoked async function expression 模式来等待fit()调用完成,然后继续后续操作
/**
 * @param sizeMB
 * @param timeSec
 */
// 不能同时拟合多个
(async function () {
    await model.fit(
        trainTensors.sizeMB,
        trainTensors.timeSec,
        {epochs: 10});
})();

// evaluate()方法会根据输入的样例特征和目标来计算损失函数的值,和fit()方法得出的损失值是一样的,但evaluate()方法并不会更新模型的权重.
// 因此,通过evaluate()方法来评估模型相对于测试集的性能,可以大致了解模型在未来应用程序中的表现情况.
model.evaluate(testTensors.sizeMB, testTensors.timeSec).print();

// 用训练集计算平均下载时间
tf.mean(trainData.timeSec).print();

// 平均绝对误差 mean absolute error: 预测值与实际值的差值的绝对值
tf.mean(tf.abs(tf.sub(testData.timeSec, 0.295))).print();

// 信息栏2-1 链式API
// 链式API模式
// const mean = testData.timeSec.sub(0.295).abs().mean();
// mean.print();

// 平均西在时间约为0.295秒,对应的误差更小.也就说.直接预测平均下载时间比模型预测更为准确.这意味着当前模型的准确率低于最简单的预测方法!
// 模型还有改进空间吗?当然,我们训练的轮次还不够多.漆面提到,.在训练过程中给,和核偏差的值一步步更新的,在这里,每个伦茨就是一部,参数值在有限的训练轮次(步骤)里可能还没达到最优点.接下来多训练几个伦茨,再来看看结果:

// 看起来之前的模型是欠拟合(underfitting)的,也就是还不哦股使用训练集.现在计算的误差低于0.05秒,大约是直接预测平均下载时间的准确率的4倍.本书提供了一些关于比庙前你和的建议,同时也会如何避免过拟合(overfitting).
// 过拟合问题更难以发现,亚视纸模型针对训练集的调整那个过多,导致不能很好地将训练规则发话到未曾见过的数据上的情况
model.evaluate(testTensors.sizeMB, testTensors.timeSec).print();


// 2.1.6 用经过训练的模型进行预测

// 借助模型的predict()方法
const smallFileMB = 1;
const bigFileMB = 100;
const hugeFileMB = 10000;
model.predict(tf.tensor2d([[smallFileMB],[bigFileMB], [hugeFileMB]])).print();


// 代码清单2-6　定义、训练、评估和预测模型
model.predict(tf.tensor2d([[7.8]])).print();


// 2.2 model.fit()



