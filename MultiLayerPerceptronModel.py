from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes= 10

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x,weights,biases):
    # Hidden layer 1 with relu activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer 2 with relu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
cost_history = []
accuracy_history = []

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost], feed_dict={x:batch_x,y:batch_y})
            avg_cost += c/batch_size
        if epoch % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            acu_temp = accuracy.eval({x:mnist.test.images, y: mnist.test.labels})
            accuracy_history.append(acu_temp)
            cost_history.append(avg_cost)
            print("Epoch:", '%04d' % (epoch+1), "- cost=", "{:.9f}".format(avg_cost), "- Accuracy=", acu_temp)

    print("Optimization Finished!")
    plt.plot(cost_history)
    plt.show()
    plt.plot(accuracy_history)
    plt.show()

    # Test the model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))









