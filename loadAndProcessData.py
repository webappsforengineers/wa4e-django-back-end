import scipy.io
import numpy as np

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
    
    for i in range(len(processed_data)):
        processed_data['input'][i][0] = np.append(processed_data['input'][i][0], processed_data['input'][i][1])
        
    processed_data['input'] = processed_data['input'][:,0]

    return processed_data

processed_data = preprocess_data(properties, curves)

print(list(processed_data.keys()))

np.info(processed_data['input'])
np.info(processed_data['output'])

print(processed_data['input'][0])

