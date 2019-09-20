import tensorflow as tf

# load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalize data
training_images = training_images / 255.0
test_images = test_images / 255.0

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train
model.fit(training_images, training_labels, epochs=5)

# test
model.evaluate(test_images, test_labels)

# predict
classifications = model.predict(test_images)

print(classifications[0])
