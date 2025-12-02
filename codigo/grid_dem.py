import numpy as np

def limit_calculation(demodulation_dictionary: dict):
    """
    Calculates the middle point between each coordinate and the two (one in case of the edges) closest coordinates in the 
    same axis.

    It divides each complex value of the demodulation dictionary into its real and imaginary components, it eliminates the 
    duplicates and sorts the list in order to figure the closest points, after that it calculates the middle point between 
    the sorted coordinates components by ading them together and dividing them by 2.

    Parameters:
        demodulation_dictionary (dict): The demodulation dictionary "integer: complex".

    Returns:
        list: the intermidiate point for the coordinates in the "I" (x) axis.
        list: the intermidiate point for the coordinates in the "Q" (y) axis.
    """

    points_I = []
    points_Q = []
    limits_I = []
    limits_Q = []
    
    for i in demodulation_dictionary.keys():
        points_I.append(demodulation_dictionary[i].real)
        points_Q.append(demodulation_dictionary[i].imag)
    points_I = list(dict.fromkeys(points_I))
    points_Q = list(dict.fromkeys(points_Q))
    points_I.sort()
    points_Q.sort()
    
    for i in range (0, len(points_I)-1):
        limits_I.append((points_I[i]+points_I[i+1])/2)
    
    for i in range (0, len(points_Q)-1):
        limits_Q.append((points_Q[i]+points_Q[i+1])/2)

    return limits_I, limits_Q, points_I, points_Q


def demodulate(data_frame: np.ndarray, demodulation_dictionary: dict):
    """
    Demodulates a given data frame based on the provided dictionary.

    It calculates the middle point betwen all the posible coordinates of the demodulation dictionary and then compares the "Q"
    and "I" values of each node of the data frame to asign the demodulated value to an ordered list in order to add it to the
    dataframe.

    Parameters:
        data_frame (np.ndarray): The dataframe that should be demodulated
        demodulation_dictionary (dict): The demodulation dictionary "integer: complex"

    Returns:
        list: The ordered demodulated values of the dataframe.
    """

    MOD_DICT = demodulation_dictionary

    limit_I_iterator = -1
    limit_Q_iterator = -1
    integer_values = []
    limits_I, limits_Q, points_I, points_Q = limit_calculation(MOD_DICT)

    for data_frame_iterator in range (0, data_frame.shape[0]):
        flag = True
        limit_I_iterator = -1
        limit_Q_iterator = -1
        while limit_I_iterator < len(limits_I)-1 and flag:
            limit_I_iterator = limit_I_iterator+1
            if data_frame["I"].loc[data_frame.index[data_frame_iterator]] <= limits_I[limit_I_iterator]:
                limit_Q_iterator = -1
                while limit_Q_iterator < len(limits_Q)-1 and flag:
                    limit_Q_iterator = limit_Q_iterator + 1
                    if data_frame["Q"].loc[data_frame.index[data_frame_iterator]] <= limits_Q[limit_Q_iterator]:
                        for point_I_iterator in range (0, len(points_I)):
                            if limits_I[limit_I_iterator] < points_I[point_I_iterator]:
                                for point_Q_iterator in range (0, len(points_Q)):
                                    if limits_Q[limit_Q_iterator] < points_Q[point_Q_iterator]:
                                        for key, val in MOD_DICT.items():
                                            if val == complex(points_I[point_I_iterator-1],points_Q[point_Q_iterator-1]):
                                                integer_values.append(key)
                                                flag = False
                                                break
                                        break
                                break
                for point_I_iterator in range (0, len(points_I)):
                    if limits_I[limit_I_iterator] < points_I[point_I_iterator]:
                        for key, val in MOD_DICT.items():
                            if val == complex(points_I[point_I_iterator-1],points_Q[len(points_Q)-1]):
                                if flag:
                                    integer_values.append(key)
                                flag = False
                                break
                        break
        limit_Q_iterator = -1
        while limit_Q_iterator < len(limits_Q)-1 and flag:
            limit_Q_iterator = limit_Q_iterator + 1
            if data_frame["Q"].loc[data_frame.index[data_frame_iterator]] <= limits_Q[limit_Q_iterator]:
                for point_Q_iterator in range (0, len(points_Q)):
                    if limits_Q[limit_Q_iterator] < points_Q[point_Q_iterator]:
                        for key, val in MOD_DICT.items():
                            if val == complex(points_I[len(points_I)-1],points_Q[point_Q_iterator-1]):
                                integer_values.append(key)
                                flag = False
                                break
                        break
        for key, val in MOD_DICT.items():
                            if val == complex(points_I[len(points_I)-1],points_Q[len(points_Q)-1]):
                                if flag:
                                    integer_values.append(key)
                                flag = False
                                break
    return integer_values
