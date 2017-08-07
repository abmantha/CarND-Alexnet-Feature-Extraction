import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

epochs = 10
BATCH_SIZE = 128

# TODO: Load traffic signs data.
with open("./train.p", mode='rb') as f: 
    data = pickle.load(f)

X_data, y_data = data['features'], data['labels']
n_classes = len(set(y_data))
# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (32, 32, 3))
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)
fc8w = tf.Variable(tf.truncated_normal(shape=shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(n_classes))
fc8 = tf.add(tf.matmul(fc8w, fc7), fc8b)
logits = tf.nn.softmax(fc8)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8w, fc8b])

predictions = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

# def evaluate(X_data, y_data): 
#     num_examples = len(X_data)
#     total_accuracy = 0
#     session = tf.get_default_session()
#     for offset in range(0, num_examples, BATCH_SIZE):
#         batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#         accuracy = session.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
#         total_accuracy += (accuracy * len(batch_x))
#     return total_accuracy / num_examples

def evaluate(X_data, y_data, sess): 
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size): 
        end = offset + batch_size
        X_batch = X_data[offset:end]
        y_batch = y_data[offset:end]
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])
    return total_loss/X_data.shape[0], total_acc/X.shape[0]

# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
with tf.Session() as session: 
    session.run(init)

    print ("Training...")
    print ()
    for i in range(epochs): 
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE): 
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_loss, validation_accuracy = evaluate(X_test, y_test, session)

            print ("EPOCH {} ...".format(i + 1))
            print ("Time %.3f seconds" % (time.time() - t0))
            print ("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print ("Validation Loss =", validation_loss)
            print ()            
