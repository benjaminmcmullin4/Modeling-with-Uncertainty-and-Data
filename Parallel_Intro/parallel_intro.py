# iPyParallel - Intro to Parallel Programming

# Benj McMullin
# Math 403
# 11/28/2023

from ipyparallel import Client
import time
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    # Initialize the client
    client = Client(connection_file='c:\\Users\\benja\\.ipython\\profile_default\\security\\ipcontroller-client.json')

    # Make a DirectView called dirview w/all engines available 
    dirview = client[:]

    # Make sure blocking is set to True
    dirview.block = True 

    # Import on all engines the scipy.sparse package
    dirview.execute("import scipy.sparse as sparse")
    client.close()

    return dirview 

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    # Initialize the client
    client = Client() 

    # Make a DirectView called dirview w/all engines available
    dirview = client[:]

    # Make sure blocking is set to True
    dirview.block = True

    # Make sure the variables are distributed
    dirview.push(dx)
    
    for i in dx.keys():

        # Get the values from the engines
        eng_val = dirview.pull(i) 
    
        for val in eng_val:
            if val != dx[i]:
                raise ValueError(f"Variable {i} has changed")

    client.close()
    return dirview

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    # Initialize lists to store results
    means, mins, maxs = [], [], [] 

    # Initialize the client
    client = Client() 

    # Make a DirectView called dirview w/all engines available
    dirview = client[:] 

    # Make sure blocking is set to True
    dirview.block = True

    # Make sure these things are imported on all engines
    dirview.execute("import numpy as np")
    dirview.execute(f"draws = np.random.normal(size={n})")

    # Calculate the mean, min, and max
    dirview.execute("mean = np.mean(draws)")
    dirview.execute("min = np.min(draws)")
    dirview.execute("max = np.max(draws)")

    # Gather the results
    means = dirview.gather("mean")
    mins = dirview.gather("min")
    maxs = dirview.gather("max")
    client.close()

    return means, mins, maxs

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    # Check number of cores
    client = Client()
    num_cores = client.ids
    client.close()

    # Store times
    para = []
    seri = []
    times = [1000000, 5000000, 10000000, 15000000]

    for i in times:
        # Start parallel computing
        start = time.time()
        prob3(i)
        end = time.time()
        para.append(end - start)

        # Start serial computing
        start = time.time()
        for j in range(len(num_cores)):
            draws = np.random.normal(size = i)
        end = time.time()
        seri.append(end - start)
    client.close(client)
    # Plot the results
    plt.plot(times, para, label = "This One's Parallel")
    plt.plot(times, seri, label = "This One's Serial")
    plt.xlabel("n")
    plt.ylabel("Time")
    plt.title("Time comparison of parallel and serial computing")
    plt.legend()
    plt.show()

# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # Initialize the client
    client = Client()
    
    # Make a DirectView called dirview w/all engines available
    dirview = client[:]

    # Make sure blocking is set to True
    dirview.block = True
    
    # Get the points
    pts = np.linspace(a, b, n)
       
    # Distribute the function and the points
    dirview.push({'f':f, 'a':a, 'b':b, 'n':n})
    dirview.scatter("points", pts[:-1])

    # Calculate the integral
    dirview.execute("value = 0")
    dirview.execute("for i in range(len(points)): value += (f(points[i]) + f(points[i] + ((b-a)/(n-1)) ))")
    dirview.execute("value *= (b-a)/(2*(n-1))")
    results = sum(dirview.gather("value"))

    client.close()
    return results

if __name__ == '__main__':
    # Problem 1
    dirview = prob1()
    print("Problem 1:")
    print(dirview)

    # Problem 2
    # dx = {'a': 10, 'b': 5, 'c': 2}
    # dirview = variables(dx)
    # print("Problem 2:")
    # print(dirview)

    # Problem 3
    # means, mins, maxs = prob3()
    # print("Problem 3:")
    # print(f"Means: {means}")
    # print(f"Mins: {mins}")
    # print(f"Maxs: {maxs}")

    # Problem 4
    #prob4()

    # Problem 5
    # a = 0
    # b = 1
    # f = lambda x: x
    # n = 200

    # print(parallel_trapezoidal_rule(f, a, b, n))
    pass