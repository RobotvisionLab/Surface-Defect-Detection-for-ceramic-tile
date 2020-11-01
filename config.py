#一些宏定义
IMAGE_SIZE=[1408,512]
POSITIVE_KolektorSDD = [[str(i) for i in range(98, 1492)]]*50
#计算分割网络的precision，accuracy,recall
#分割网络输出像素的概率大于该阈值则认为是缺陷（白点）
SEGMENT_OUTPUT_THRESHOLD = 0
csv_name = 'threshold_%s_.csv' % str(SEGMENT_OUTPUT_THRESHOLD)

