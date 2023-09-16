import pickle

with open("./simulation_parameters.p", "rb") as openfile:
    intensities = pickle.load(openfile)
    print(intensities)