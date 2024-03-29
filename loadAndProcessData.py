import scipy.io
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models, optimizers, regularizers
import datetime


# Load data from .mat data file
def load_data(filename):
    mat = scipy.io.loadmat(filename)

    properties = mat["properties"]
    curves = mat["curves"]
    return properties, curves

# shuffle inputs and outputs in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.default_rng(seed=43).permutation(len(a))
    return a[p], b[p]

# Function to expand arrays into multiple columns
def expand_array(row):
    return pd.Series(row['column_of_arrays']) 

def preprocess_data(properties, curves):
    processed_data = {'input': np.empty((0, 2)), 'output': np.empty((0, 1))}
    
    for j in range(len(properties)):
        for k in range(len(properties[j][0])):
            for l in range(curves[j][0][k][0].shape[0]):
                processed_data['input'] = np.vstack((processed_data['input'], np.hstack((properties[j][0][k].T, curves[j][0][k][0][l, 0]))))
                processed_data['output'] = np.vstack((processed_data['output'], curves[j][0][k][0][l, 1]))
    
    for i in range(len(processed_data['input'])):
        processed_data['input'][i][0] = np.append(processed_data['input'][i][0], processed_data['input'][i][1])
        
    processed_data['input'] = processed_data['input'][:,0]
    
    shuffled_data = unison_shuffled_copies(processed_data['input'], processed_data['output'])
    
    # reshape input data
    inputs_array = pd.DataFrame(shuffled_data[0])
    inputs_array.columns = ['column_of_arrays']
    inputs = inputs_array.apply(expand_array, axis=1)
    targets = shuffled_data[1]

    return shuffled_data, inputs, targets

def validate_data():
    print('validate_data')

def filter_data(shuffled_data,
                p_min, p_max, 
                ppa_min, ppa_max, 
                ocr_min, ocr_max, 
                e_min, e_max,
                dr_min, dr_max,
                d50_min, d50_max,
                cu_min, cu_max,
                g0_min, g0_max):

    
    filtered_inputs = []
    filtered_targets = []

    for i in range(len(shuffled_data[0])):
        if shuffled_data[0][i][0] >= p_min and shuffled_data[0][i][0] <= p_max and \
        shuffled_data[0][i][1] >= ppa_min and shuffled_data[0][i][1] <= ppa_max and \
        shuffled_data[0][i][2] >= ocr_min and shuffled_data[0][i][2] <= ocr_max and \
        shuffled_data[0][i][3] >= e_min and shuffled_data[0][i][3] <= e_max and \
        shuffled_data[0][i][4] >= dr_min and shuffled_data[0][i][4] <= dr_max and \
        shuffled_data[0][i][5] >= d50_min and shuffled_data[0][i][5] <= d50_max and \
        shuffled_data[0][i][6] >= cu_min and shuffled_data[0][i][6] <= cu_max and \
        shuffled_data[0][i][7] >= g0_min and shuffled_data[0][i][7] <= g0_max:
            filtered_inputs.append(shuffled_data[0][i])
            filtered_targets.append(shuffled_data[1][i])    

    filtered_inputs = np.asarray(filtered_inputs)
    filtered_targets = np.asarray(filtered_targets)
    
    return filtered_inputs, targets


def plot_loss(history, fig_name):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)
    plt.clf()
    
def plot_output_v_target(targets, outputs, fig_name):
    print(targets.shape)
    print(outputs.shape)
    # p = np.polyfit(targets[0], outputs[0], 1)
    # print(f'Fit: y = {str(round(p[0],5))} x + {str(round(p[1], 5))}')
    # fit_equation = f'Fit: y = {str(round(p[0],5))} x + {str(round(p[1], 5))}'
    plt.figure()
    plt.scatter(targets, outputs)
    # plt.plot(targets, p[0]*targets+p[1], color='black', linewidth=2)
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    # plt.text(2, 0.65, str(p))
    # plt.title(fit_equation)
    plt.savefig(fig_name)
    plt.clf()
    
def plot_strain_v_GGo(strain, GGo, fig_name):
    plt.figure()
    plt.scatter(strain, GGo)
    plt.xscale('log')
    plt.xlabel('Strain')
    plt.ylabel('G/G0')
    plt.title('Output Curve')
    plt.savefig(fig_name)
    plt.clf()
    
def divide_dataset(train_ratio, val_ratio, inputs, targets):
    # Set up Division of Data for Training, Validation, Testing

    num_samples = inputs.shape[0]
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    x_train, y_train = inputs[0:num_train,:], targets[0:num_train, :]
    x_val, y_val = inputs[num_train:num_train + num_val, :], targets[num_train:num_train + num_val, :]
    x_test, y_test = inputs[num_train + num_val:, :], targets[num_train + num_val:, :]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def train_model(inputs, targets, propSelect):
    
    inputs_df = pd.DataFrame(inputs) 
    selected_inputs = inputs_df[np.append(propSelect, 8)]
    inputs = selected_inputs.to_numpy()

    # Construct a neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(propSelect) + 1,)),
        tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(propSelect) + 1,)),
        # tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(propSelect) + 1,)),
        # tf.keras.layers.Dropout(0.05), # Adjust dropout rate as needed
        tf.keras.layers.Dense(units=1, activation='linear')  # Output layer
    ])

    # print(model.layers)
    # print(model.summary())

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Nadam(), loss='mse')
    
    # log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # learning rate scheduler
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_delta=0.0001, min_lr=0.00001)
    
    x_train, y_train, x_val, y_val, x_test, y_test = divide_dataset(0.7, 0.15, inputs, targets)

    # Train the model
    history = model.fit(x_train, y_train, epochs=150, validation_data=(x_val, y_val), verbose=0, batch_size=16, callbacks=[lr_callback])
    
    # # callbacks argument when using tensorboard
    # callbacks=[tensorboard_callback, lr_callback]

    # Plot how the loss (MSE) changed throughout the epochs
    plot_loss(history, 'plot_loss.png')

    performance = model.evaluate(x_test, y_test)

    outputs = model.predict(inputs)
    plot_output_v_target(targets, outputs, 'regression_plot.png')
    
    return performance, model

def predict_GGo(properties, propSelect, model):
    # Calculate GGo for different strain values
    strain = np.logspace(-4, 1, 200)
    GGo = []

    for s in strain:
        inputs_prop = np.append(properties[propSelect], s).reshape(1, -1)
        GGo.append(model.predict(inputs_prop)[0][0])

    # Create a table (pandas DataFrame) with strain and GGo values
    data = {'Strain': strain, 'GGo': GGo}
    output_data = pd.DataFrame(data)
    
    return output_data, strain, GGo
    

properties, curves = load_data('StiffnessNNAppData.mat')
shuffled_data, inputs, targets = preprocess_data(properties, curves)

# Define properties and propSelect
propSelect = [0, 1, 2, 3, 4, 5, 6, 7]  
properties = np.array([0.1, 1.019, 1, 0.76, 0.6923, 0.99, 1.2, 161.1])

    
performance, model = train_model(inputs, targets, propSelect)

output_data, strain, GGo = predict_GGo(properties, propSelect, model)
plot_strain_v_GGo(strain, GGo, 'output_curve.png')

print(f'performance: {performance}')
# print(output_data)