import pickle
import debugpy
import numpy
import os

# debugpy.connect(('192.168.1.50', 6789))
# debugpy.wait_for_client()
# debugpy.breakpoint()

for i in range(4):
    result = pickle.load(open(f"result{i + 1}.pkl", "rb"))
    likelihood, fited_alpha, fited_mu = result

    for j in range(fited_alpha.shape[0]):
        for k in range(fited_alpha.shape[1]):
            if fited_alpha[j][k] != 0:
                fited_alpha[j][k] = 1

    if os.path.exists(f"datasets/dataset_{i+1}/causal_prior.npy"):
        prior_matrix = numpy.load(f"datasets/dataset_{i+1}/causal_prior.npy")
        assert fited_alpha.shape == prior_matrix.shape
        for j in range(fited_alpha.shape[0]):
            for k in range(fited_alpha.shape[0]):
                if prior_matrix[j][k] == 1:
                    fited_alpha[j][k] = 1
                if prior_matrix[j][k] == 0:
                    fited_alpha[j][k] = 0
    
    numpy.fill_diagonal(fited_alpha, 0)


    numpy.save(f"dataset_{i+1}_graph_matrix.npy", fited_alpha)
