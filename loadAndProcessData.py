import scipy.io
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from .mat data file
mat = scipy.io.loadmat('StiffnessNNAppData.mat')

# Examine the keys of the data
# print(list(mat.keys()))

properties = mat["properties"]
curves = mat["curves"]

# print(type(properties))
# print(type(curves))

# np.info(properties)
# np.info(curves)

# print(properties[0][0][0])

# print(len(properties))
# print(len(properties[0]))
# print(len(properties[0][0]))
# print(len(properties[0][0][0][0]))


def preprocess_data(properties, curves):
    processed_data = {'input': np.empty((0, 2)), 'output': np.empty((0, 1))}
    loaded_count = 0
    filtered_count = 0
    final_count = 0
    
    for j in range(len(properties)):
        for k in range(len(properties[j][0])):
            for l in range(curves[j][0][k][0].shape[0]):
                processed_data['input'] = np.vstack((processed_data['input'], np.hstack((properties[j][0][k].T, curves[j][0][k][0][l, 0]))))
                processed_data['output'] = np.vstack((processed_data['output'], curves[j][0][k][0][l, 1]))
                loaded_count += 1
                final_count += 1
            
            filtered_count += curves[j][0][k][0].shape[0]
            loaded_count += curves[j][0][k][0].shape[0]
        
        filtered_count += curves[j][0][k][0].shape[0]
        loaded_count += curves[j][0][k][0].shape[0]
    
    for i in range(len(processed_data['input'])):
        processed_data['input'][i][0] = np.append(processed_data['input'][i][0], processed_data['input'][i][1])
        
    processed_data['input'] = processed_data['input'][:,0]

    return processed_data

processed_data = preprocess_data(properties, curves)



inputs_array = pd.DataFrame(processed_data['input'])
inputs_array.columns = ['column_of_arrays']
# print(inputs)

# Function to expand arrays into multiple columns
def expand_array(row):
    return pd.Series(row['column_of_arrays']) 

inputs_df = inputs_array.apply(expand_array, axis=1)
# print(inputs_df.shape)
# print(inputs_df.head())
# Define properties and propSelect
propSelect = [0, 1, 2, 3, 4, 5, 6, 7]  # Python indices start from 0
properties = np.array([0.1, 1.019, 1, 0.76, 0.6923, 0.99, 1.2, 161.1])
layers = [12]

# # Get inputs from processed data
selected_inputs = inputs_df[np.append(propSelect, 8)]


inputs = selected_inputs.to_numpy()
print(inputs.shape)

# # Get outputs from processed data
targets = processed_data['output']
# print(targets.shape)


# # Construct a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=12, activation='tanh', input_shape=(len(propSelect) + 1,)),
    tf.keras.layers.Dense(units=1)  # Output layer
])

# print(model.layers)
# print(model.summary())

# # Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Set up Division of Data for Training, Validation, Testing
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_samples = inputs.shape[0]
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)


x_train, y_train = inputs[0:num_train,:], targets[0:num_train,:]
print(type(x_train))
print(x_train.shape)
print(type(y_train))
print(y_train.shape)
# print(x_train)
# print(y_train)
x_val, y_val = inputs[num_train:num_train + num_val, :], targets[num_train:num_train + num_val,:]
print(x_val.shape)
print(type(x_val))
print(y_val.shape)
print(type(y_val))

x_test, y_test = inputs[num_train + num_val:, :], targets[num_train + num_val:, :]
# print(x_test.shape)
# print(y_test.shape)

def plot_loss(history, fig_name):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)
    plt.clf()

# print(x_train.shape)
# print(x_train[0:5])
# print(y_train.shape)
# print(y_train[0:5])

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))
# validation_data=(x_val, y_val)
# print(history.history)

plot_loss(history, 'history.png')
# # Test the model
# outputs = model.predict(inputs)
# performance = model.evaluate(inputs, targets, verbose=0)

# # Plot regression
# plt.figure()
# plt.plot(targets, outputs, 'o')
# plt.xlabel('True values')
# plt.ylabel('Predicted values')
# plt.title('Regression plot')
# plt.show()

# # Calculate GGo for different strain values
# strain = np.logspace(-4, 1, 200)
# GGo = []

# for s in strain:
#     inputs_prop = np.append(properties[propSelect], s).reshape(1, -1)
#     GGo.append(model.predict(inputs_prop)[0][0])

# # Create a table (pandas DataFrame) with strain and GGo values
# data = {'Strain': strain, 'GGo': GGo}
# b = pd.DataFrame(data)