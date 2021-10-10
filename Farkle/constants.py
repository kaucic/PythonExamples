import json

class CONST:
    MAX_RECURSION_DEPTH = 3
    
    CUTOFF_E13 = float("inf")
    CUTOFF_E23 = float("inf")
    CUTOFF_E33 = float("inf")

    try:
        with open("data/constants.json") as file_stream:
            file_data = file_stream.read()

        json_object = json.loads(file_data)

        MAX_RECURSION_DEPTH = \
            json_object.get("MAX_RECURSION_DEPTH") or MAX_RECURSION_DEPTH

        CUTOFF_E13 = json_object.get("CUTOFF_E13") or CUTOFF_E13
        CUTOFF_E23 = json_object.get("CUTOFF_E23") or CUTOFF_E23
        CUTOFF_E33 = json_object.get("CUTOFF_E33") or CUTOFF_E33
    
    except FileNotFoundError:
        # If there is no file, use the defaults.
        pass