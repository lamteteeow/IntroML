from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
from scipy.signal import chirp as scichirp
import numpy as np
import matplotlib.pyplot as plt

def main():
#    ################################################
#    ## Linear Chirp
#    ################################################
#
#    linchirp = createChirpSignal(200, 1, 1, 10, True)
#
#    t = np.linspace(0, 1, 200)
#    scilin = scichirp(t, f0=1, f1=10, t1=1, method='linear')
#
#    # Plot Scipy Chirp
#    plt.subplot(2, 2, 1)
#    plt.plot(t, scilin)
#    plt.title("Linear Chirp - Scipy, f(0)=1, f(10)=10")
#    plt.subplot(2, 2, 3)
#
#    # Plot Selfmade Chirp
#    plt.plot(t, linchirp)
#    plt.title("Linear Chirp - Selfmade")
#    plt.xlabel('t (sec)')
#
#    ################################################
#    ## Exponential Chirp
#    ################################################
#
#    expchirp = createChirpSignal(200, 1, 1, 10, False)
#
#    t = np.linspace(0, 1, 200)
#    sciexp = scichirp(t, f0=1, f1=10, t1=1, method='quadratic')
#
#    # Plot Scipy Chirp
#    plt.subplot(2, 2, 2)
#    plt.plot(t, sciexp)
#    plt.title("Exp. Chirp - Scipy, f(0)=1, f(10)=10")
#
#    # Plot Selfmade Chirp
#    plt.subplot(2, 2, 4)
#    plt.plot(t, expchirp)
#    plt.title("Exp. Chirp - Selfmade")
#    plt.xlabel('t (sec)')
#    
#    plt.show()

    ################################################
    ## Decomposition: Triangle Signal
    ################################################

    triangle = createTriangleSignal(200, 2, 10000)
    t = np.linspace(0, 1, 200)
    # Plot Triangle Signal
    plt.subplot(3, 1, 1)
    plt.plot(t, triangle)
    plt.title("Triangle Signal")
    plt.xlabel('t (sec)')
        
    ################################################
    ## Decomposition: Square Signal
    ################################################

    square = createSquareSignal(200, 2, 10000)
    # Plot Triangle Signal
    plt.subplot(3, 1, 2)
    plt.plot(t, square)
    plt.title("Square Signal")
    plt.xlabel('t (sec)')    
    
    ################################################
    ## Decomposition: Sawtooth Signal
    ################################################

    sawtooth = createSawtoothSignal(200, 2, 10000, 1)
    # Plot Triangle Signal
    plt.subplot(3, 1, 3)
    plt.plot(t, sawtooth)
    plt.title("Sawtooth Signal")
    plt.xlabel('t (sec)')
    
    plt.show()    

    
if __name__=="__main__":
    main()
