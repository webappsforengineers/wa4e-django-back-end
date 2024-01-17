import scipy.io
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models, optimizers, regularizers


# Load data from .mat data file
mat = scipy.io.loadmat('StiffnessNNAppData.mat')

properties = mat["properties"]
curves = mat["curves"]

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

# shuffle inputs and outputs in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


shuffled_data = unison_shuffled_copies(processed_data['input'], processed_data['output'])

inputs_array = pd.DataFrame(shuffled_data[0])
inputs_array.columns = ['column_of_arrays']

# Function to expand arrays into multiple columns
def expand_array(row):
    return pd.Series(row['column_of_arrays']) 

inputs_df = inputs_array.apply(expand_array, axis=1)

# Define properties and propSelect
propSelect = [0, 1, 2, 3, 4, 5, 6, 7]  
properties = np.array([0.1, 1.019, 1, 0.76, 0.6923, 0.99, 1.2, 161.1])
layers = [12]

# Get inputs from processed data
selected_inputs = inputs_df[np.append(propSelect, 8)]

inputs = selected_inputs.to_numpy()

# Get outputs from processed data
targets = shuffled_data[1]

# Construct a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(propSelect) + 1,)),
    tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(propSelect) + 1,)),
    tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(propSelect) + 1,)),
    tf.keras.layers.Dropout(0.6), # Adjust dropout rate as needed
    tf.keras.layers.Dense(units=1, activation='linear')  # Output layer
])

# print(model.layers)
# print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Set up Division of Data for Training, Validation, Testing
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_samples = inputs.shape[0]
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)


x_train, y_train = inputs[0:num_train,:], targets[0:num_train, :]
x_val, y_val = inputs[num_train:num_train + num_val, :], targets[num_train:num_train + num_val, :]
x_test, y_test = inputs[num_train + num_val:, :], targets[num_train + num_val:, :]



# # Train the model
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), verbose=0)

def plot_loss(history, fig_name):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)
    plt.clf()


plot_loss(history, 'plot_loss.png')

performance = model.evaluate(x_test, y_test)
print(f'performance: {performance}')

outputs = model.predict(inputs)

plt.figure()
plt.scatter(targets, outputs)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig('targets_vs_outputs.png')
plt.clf()

# Calculate GGo for different strain values
strain = np.logspace(-4, 1, 200)
GGo = []

for s in strain:
    inputs_prop = np.append(properties[propSelect], s).reshape(1, -1)
    GGo.append(model.predict(inputs_prop)[0][0])

plt.figure()
plt.scatter(strain, GGo)
plt.xscale('log')
plt.xlabel('Strain')
plt.ylabel('G/G0')
plt.title('Output Curve')
plt.savefig('output_curve.png')
plt.clf()

# # Create a table (pandas DataFrame) with strain and GGo values
# data = {'Strain': strain, 'GGo': GGo}
# b = pd.DataFrame(data)