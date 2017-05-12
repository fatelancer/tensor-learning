
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# 根据定义的形状随机初始化一个变量, 用于当作CNN的权值矩阵
def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name:
      return tf.Variable(initial, name=name)
  return tf.Variable(initial)

# 这个用于当作偏移量, 就是公式中的X0.
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积函数模板
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max pooling的池化模板
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():
    mnist = read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()

    # x为输入占位符, y_为输出占位符, y为正确输出占位符
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    y = tf.placeholder("float", shape=[None, 10])

    # 第一层卷积，卷积在每个5x5的patch中算出32个特征
    W_conv1 = weight_variable([5, 5, 1, 32], name="W1")
    b_conv1 = bias_variable([32])
    # 为了使用这一层，将x转变维度为4维，第一维度不定，表示batchsize, 第二三维度是高和宽，
    # 第四维度是通道数
    x_image = tf.reshape(x, [-1,28,28,1])
    # 卷积和池化
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 密集连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 使用dropout防止过拟合
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 保存模型
    if not tf.gfile.Exists("../asset/mnist_cnn/"):
        tf.gfile.MakeDirs("../asset/mnist_cnn/")

    # 定义后开始训练和评估模型
    print("define over, training start ...")
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(201):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))



        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print(sess.run(tf.get_default_graph().get_tensor_by_name("W1:0")))

    saver = tf.train.Saver({"W1": W_conv1, "B1": b_conv1})
    saver.save(sess, "../asset/mnist_cnn/mnist",  meta_graph_suffix="model")





# 模型加载直接使用
def test():
    return


print(__name__)
main()
#test()