
import math
import numpy as np

def calculate_raw_amplitudes_and_phases(csi_data):

    amplitudes = []
    phases = []

    i = 0
    while i != len(csi_data):
        
        img = int(csi_data[i])
        real = int(csi_data[i+1])

        amplitude  = math.sqrt(img**2 + real**2)
        amplitudes.append(amplitude)

        # manage division by zero error 
        if real == 0:
            phases.append(0)
            i += 2
            continue

        phase = np.arctan2(img, real)
        phases.append(phase)

        i += 2

    return amplitudes, phases

