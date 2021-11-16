// 代码清单2-9 数据标准化:平均值为0,标准差为单位标准差
/**
 * 计算数组中每列数据的平均值和标准差
 * @param data: 用于独立计算每列数据的平均值和标准差的数据集
 * @returns  包含每列数据的平均值和标准差的一维张量
 */
export function determineMeanAndStdDev (data) {
  const dataMean = data.mean(0)
  const diffFromMean = data.sub(dataMean)
  const squaredDiffFromMean = diffFromMean.square()
  const variance = squaredDiffFromMean.mean(0)
  const std = variance.sqrt()
  return { mean, std }
}

/**
 * 输入给定的平均值和标准差,通过减去平均值并除以标准差,实现数据标准化
 * @param data: 待标准化的数据,形状为[numSamples, numFeatures]
 * @param dataMean: 输入的数据平均值,形状为[numFeatures]
 * @param dataStd: 输入的数据的标准差,形状为[numFeatures]
 * @returns {*}: 返回的张量和输入的数据的形状相同,但通过标准化,每列数据的平均值变为零,标准差变为单位标准差
 */
export function normalizeTensor (data, dataMean, dataStd) {
  return data.sub(dataMean).div(dataStd)
}
