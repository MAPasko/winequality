# получение данных
import data

(train_data, train_quality), (test_data, test_quality) = data.take_data()

# нормализация данных
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

from tensorflow.keras.utils import to_categorical

train_quality = to_categorical(train_quality)
test_quality = to_categorical(test_quality)

print(train_quality)
print(test_quality)

# определение модели
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

# обучение модели
history = model.fit(train_data, train_quality,
                    epochs=150, batch_size=128)

# график потерь на этапах обучения и проверки
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# график точности на этапах обучения и проверки
plt.clf()
acc = history.history["accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# проверка обученной модели
results = model.evaluate(test_data, test_quality)

