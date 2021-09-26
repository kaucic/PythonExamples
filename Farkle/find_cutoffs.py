import expected_values
import json
import os
import scipy.optimize as optim

def find_cutoff(callable):
    def f(x):
        return callable(x, counter=0) - x

    result = optim.root_scalar(f, method="bisect", bracket=[0, 1_000_000])

    return result.root

if __name__ == "__main__":
    functions = {
        "E13": expected_values.compute_E13_total_recursive,
        "E23": expected_values.compute_E23_total_recursive,
        "E33": expected_values.compute_E33_total_recursive,
    }

    data = dict()
    with open("data/constants.json", "r") as input_stream:
        data = json.load(input_stream)
    
    for turn_type in functions:
        key = "CUTOFF_" + turn_type
        value = find_cutoff(functions[turn_type])
        data[key] = value
    
    os.makedirs("data", exist_ok=True)
    with open("data/constants.json", "w") as output_stream:
        json.dump(data, output_stream)