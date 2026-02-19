# general_utils.py

import numpy as np

def get_input(prompt, default, shape=None):
    """
    Objective
    ---------
    Prompt the user for input and return it as a NumPy array.

    Input
    -----
    prompt: str
        The text shown to the user when requesting input.
    default: list, tuple, or ndarray
        Default value to use if the user presses Enter without typing anything.
    shape: tuple, optional
        Expected shape of the input array.

    Output
    ------
    arr: np.ndarray
        The input provided by the user (or default) converted to a NumPy ndarray.

    Notes
    -----
    The function checks the input array shape and raises a ValueError if it does not match the expected one.
    """
    user_input = input(prompt)
    arr = np.asarray(eval(user_input)) if user_input else np.asarray(default)
    if arr.shape != shape:
        raise ValueError(f"Invalid input shape {arr.shape}, expected {shape}.")
    return arr