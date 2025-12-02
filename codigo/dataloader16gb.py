from pathlib import Path

import numpy as np
import pandas as pd

def dataloader16gb(PATH: str, SPACING: str, OSNR: str) -> tuple[np.array, np.array]:
    """
    Loads the data to be used, only loads the relevant data.

    It searches the data base parsing the names of the directorys until it finds the one whose "GHz" matches with the
    spacing, after that it serarches that directory parcing the names of the files until it finds the one whose "db"
    matches with the OSNR then it loads the file and returns the data.

    Parameters:
        PATH (string): the relative path to the database.
        SPACING (string): the user defined spacing.
        OSNR (string): The user defined osnr.
    Returns:
        rx (np.ndarray): the I and Q coordinates of the imaginary numbers.
        tx (np.ndarray): the I and Q coordinates.
    """
    DB_Path = Path(PATH)
    for directory in sorted(DB_Path.iterdir()):
        if directory.is_file() or directory.name[-3:] != "GHz":
            if directory.name == "single_ch":
                dir_name = "50GHz"
            elif directory.name == "2x16QAM_16GBd.csv":
                tx = pd.read_csv(directory)
                continue
            else:
                continue
        else:
            dir_name = directory.name
        if SPACING == str(dir_name[0:dir_name.find("GHz")]):
            for subdir in sorted(directory.iterdir()):
                if subdir.is_dir():
                    continue
                if subdir.name[subdir.name.find("consY")+len("consY"):subdir.name.find("dB")] == OSNR:
                    rx = pd.read_csv(subdir)
    return rx, tx

