import os
import numpy as np
import tensorflow as tf
import knapsack


def brute_force_knapsack(x_weights, x_prices, x_capacity):
    item_count = x_weights.shape[0]
    picks_space = 2 ** item_count
    best_price = -1
    best_picks = np.zeros(item_count)
    for p in range(picks_space):
        picks = [int(c) for c in f"{p:0{item_count}b}"]
        price = np.dot(x_prices, picks)
        weight = np.dot(x_weights, picks)
        if weight <= x_capacity and price > best_price:
            best_price = price
            best_picks = picks
    return best_picks


def fast_knapsack(x_weights, x_prices, x_capacity):
    best_price, best_picks = knapsack.knapsack(x_weights, x_prices).solve(x_capacity)
    return best_picks


def create_knapsack(item_count=5):
    x_weights = np.random.randint(1, 45, item_count)
    x_prices = np.random.randint(1, 99, item_count)
    x_capacity = np.random.randint(1, 99)
    y = brute_force_knapsack(x_weights, x_prices, x_capacity)
    return x_weights, x_prices, x_capacity, y


def knapsack_loss(input_weights, input_prices, cvc):
    def loss(y_true, y_pred):
        picks = y_pred
        violation = tf.keras.backend.maximum(tf.keras.backend.batch_dot(picks, input_weights, 1) - 1, 0)
        price = tf.keras.backend.batch_dot(picks, input_prices, 1)
        return cvc * violation - price

    return loss


def metric_overprice(input_prices):
    def overpricing(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.batch_dot(y_pred, input_prices, 1) - tf.keras.backend.batch_dot(y_true, input_prices, 1))

    return overpricing


def metric_space_violation(input_weights):
    def space_violation(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.maximum(tf.keras.backend.batch_dot(y_pred, input_weights, 1) - 1, 0))

    return space_violation


def metric_pick_count():
    def pick_count(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.sum(y_pred, -1))

    return pick_count


def unsupervised_model(cvc=5.75, item_count=5):
    input_weights = tf.keras.Input((item_count,))
    input_prices = tf.keras.Input((item_count,))
    inputs_concat = tf.keras.layers.Concatenate()([input_weights, input_prices])
    picks = tf.keras.layers.Dense(item_count ** 2 + item_count * 2, activation="sigmoid")(inputs_concat)
    picks = tf.keras.layers.Dense(item_count, activation="sigmoid")(picks)
    model = tf.keras.Model(inputs=[input_weights, input_prices], outputs=[picks])
    model.compile("adam",
                  knapsack_loss(input_weights, input_prices, cvc),
                  metrics=[tf.keras.metrics.binary_accuracy, metric_space_violation(input_weights),
                           metric_overprice(input_prices), metric_pick_count()])
    return model


def supervised_model(item_count=5):
    input_weights = tf.keras.Input((item_count,), name="Weights")
    input_prices = tf.keras.Input((item_count,), name="Prices")
    inputs_concat = tf.keras.layers.Concatenate(name="Concatenate")([input_weights, input_prices])
    picks = tf.keras.layers.Dense(item_count ** 2 + item_count * 2, activation="sigmoid", name="Hidden")(inputs_concat)
    picks = tf.keras.layers.Dense(item_count, activation="sigmoid", name="Output")(picks)
    model = tf.keras.Model(inputs=[input_weights, input_prices], outputs=[picks])
    model.compile("adam",
                  tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy, metric_space_violation(input_weights),
                           metric_overprice(input_prices), metric_pick_count()])
    return model


def create_knapsack_dataset(count):
    x1 = []
    x2 = []
    y = []
    for _ in range(count):
        x_weights, x_prices, x_capacity, answer = create_knapsack()
        x1.append(x_weights / x_capacity)
        x2.append(x_prices / x_prices.max())
        y.append(answer)
    return [np.array(x1), np.array(x2)], np.array(y)


def train_knapsack(model, train_x, train_y, test_x, test_y):
    if os.path.exists("best_model.h5"): os.remove("best_model.h5")
    model.fit(train_x, train_y, epochs=512, verbose=1,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor="binary_accuracy", save_best_only=True,
                                                     save_weights_only=True)])
    model.load_weights("best_model.h5")
    train_results = model.evaluate(train_x, train_y, 64, 0)
    test_results = model.evaluate(test_x, test_y, 64, 0)
    print("Model results(Train/Test):")
    print(f"Loss:               {train_results[0]:.2f} / {test_results[0]:.2f}")
    print(f"Binary accuracy:    {train_results[1]:.2f} / {test_results[1]:.2f}")
    print(f"Space violation:    {train_results[2]:.2f} / {test_results[2]:.2f}")
    print(f"Overpricing:        {train_results[3]:.2f} / {test_results[3]:.2f}")
    print(f"Pick count:         {train_results[4]:.2f} / {test_results[4]:.2f}")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    train_x, train_y = create_knapsack_dataset(10000)
    test_x, test_y = create_knapsack_dataset(200)
    model = supervised_model()
    train_knapsack(model, train_x, train_y, test_x, test_y)
    model = unsupervised_model()
    train_knapsack(model, train_x, train_y, test_x, test_y)
