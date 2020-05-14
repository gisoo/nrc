import SimulationSpecification
from NetworkGraph import NetworkGraph
from time import sleep
import sys
import matplotlib.pyplot as plt
import numpy as np

simulation_specification: SimulationSpecification
network_graph: NetworkGraph


class Simulation:
    """This class representing the simulation and actin as the main class for calling other classes and methods to
    create a simulation environment or calculate and draw required graphs. """

    def __init__(self):
        self.simulation_specification = SimulationSpecification.SimulationSpecification()

    def set_simulation_specification(self):
        """Sets all the required specifications of a simulation."""
        self.simulation_specification.set_number_of_nodes()
        self.simulation_specification.set_x0()
        self.simulation_specification.set_epsilon()
        self.simulation_specification.set_min_accepted_divergence()
        return

    def generate_network_graph(self, simulation_specification: SimulationSpecification):
        """Generates an instance of network graph based on the simulation specification."""
        self.network_graph = NetworkGraph(simulation_specification)
        return

    def generate_convergence_plot(self):
        """Calculates the distance between generated xi of each iteration with expected outcome and finally plot this
        distance over simulation iteration. """
        distances = []
        x0 = np.array(simulation.network_graph.optimum_point)
        for i in range(len(simulation.network_graph.nodes[0].all_calculated_xis)):
            xi = np.array(simulation.network_graph.nodes[0].all_calculated_xis[i])
            dst = np.sqrt(np.sum((x0 - xi) ** 2))
            distances.append(dst)
        plt.plot(distances)
        plt.ylabel('Distance till optimum point')
        plt.xlabel('Iteration')
        plt.savefig('books_read.png')
        plt.show()

    def wait_until_result_founded(self) -> None:
        is_all_dif_accepted = False
        while True and not (is_all_dif_accepted):
            sleep(5)
            is_all_dif_accepted = True
            for i in range(len(simulation.network_graph.nodes)):
                if not (simulation.network_graph.nodes[i].has_result_founded()):
                    is_all_dif_accepted = False
                    break
        print("Result founded: ")


# Creating and executing the simulation based on simulation specification:
simulation: Simulation = Simulation()
simulation.set_simulation_specification()

# Generating the network graph with specified number of nodes based on the simulation specification:
simulation.generate_network_graph(simulation.simulation_specification)
simulation.network_graph.draw_graph()

# Starting the all nodes' threads in simulation graph:
for i in range(len(simulation.network_graph.nodes)):
    simulation.network_graph.nodes[i].daemon = True
    simulation.network_graph.nodes[i].start()

# Continue the simulation until all nodes reach a consensus estimation about the target xi:
simulation.wait_until_result_founded()

# Plot the distance between selected xi and target xi in different iterations:
simulation.generate_convergence_plot()

# Finish the simulation!
sys.exit()
