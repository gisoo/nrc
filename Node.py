import threading
import time
import numpy as np
from Message import *



class Node(threading.Thread):
    """Representing the node and its actions and attributes. This class inherits the Thread class, so each instance
    of this class would be executed in a separate thread. This is important in order of improving the simulation
    speed and also simulating asynchronous communications """

    def __init__(self, node_id: int, x0: np.array, epsilon: float, all_nodes_message_buffers: [],
                 minimum_accepted_divergence: float,
                 adjacency_vector: np.array,
                 function_constants: np.array, simulation_function_xtx_btx):

        super(Node, self).__init__()
        self.lock = threading.Lock()   
        self.node_id = node_id
        self.epsilon = epsilon
        self.xi = np.array(x0)
        self.all_calculated_xis = []
        self.function_constants = function_constants
        self.simulation_function_xtx_btx = simulation_function_xtx_btx

        self.all_nodes_message_buffers = all_nodes_message_buffers
        self.number_of_neighbors = np.sum(adjacency_vector)
        self.minimum_accepted_divergence = minimum_accepted_divergence
        self.adjacency_vector = adjacency_vector

        self.yi = np.zeros(x0.size)
        self.gi = np.zeros(x0.size)
        self.gi_old = np.zeros(x0.size)
        self.zi = np.eye(x0.size)
        self.hi = np.eye(x0.size)
        self.hi_old = np.eye(x0.size)

        self.is_ready_to_receive = False
        self.is_ready_to_update = False
        self.is_ready_to_transmit = True

    def run(self) -> None:
        """ This method is the start point for the executing the thread for each node. By calling the start() method
        on a node, this method will be called. """
        self.transmit_data()

    def transmit_data(self):
        """This method update the yi and zi in each iteration and create a message including the new updated yi and
        zi. Finally the new message will be broadcast to all neighbors of this node. """
        print("Node ID:  " + str(self.node_id) + "  -  transmitting data started!\n")
        if self.is_ready_to_transmit:
            self.is_ready_to_transmit = False

            self.yi = (1 / (self.number_of_neighbors + 1)) * self.yi
            self.zi = self.zi / (self.number_of_neighbors + 1)

            message = Message(self.node_id, self.yi, self.zi)
            self.broadcast(message)
            self.is_ready_to_receive = True
            print("Node ID:  " + str(self.node_id) + "  -  transmitting data ended!\n")
            self.receive_data()
        return

    def broadcast(self, message):
        """This method will broadcast the passed message to all neighbors of this node."""
        print("Node ID:  " + str(self.node_id) + "  -  broadcasting started!\n")
        with self.lock:
            i = 0
            while i < len(self.all_nodes_message_buffers):
                if self.adjacency_vector[i] == 1:
                    self.all_nodes_message_buffers[i].put(message)
                i += 1
        time.sleep(0.05)
        print("Node ID:  " + str(self.node_id) + "  -  broadcasting ended!\n")
        return

    def receive_data(self):
        """This method will handle the reception of data from neighbors. Using the yi and zi from the messages,
        the yi and zi would be updated by this method."""
        if self.is_ready_to_receive:
            print("Node ID:  " + str(self.node_id) + "  -  Receiving data started!\n")
            with self.lock:
                message = self.all_nodes_message_buffers[self.node_id].get()
                time.sleep(0.05)
            self.is_ready_to_receive = False

            self.j = message.node_id
            self.yi = self.yi + message.yi
            self.zi = self.zi + message.zi

            self.is_ready_to_update = True
            print("Node ID:  " + str(self.node_id) + "  -  Receiving data ended!\n")
            self.update_estimation()
        return

    def update_estimation(self):
        """This method will calculate the next xi and also update the hi and gi by using the new xi. """
        print("Node ID:  " + str(self.node_id) + "  -  Updating data started!\n")
        if self.is_ready_to_update:
            self.is_ready_to_update = False
            self.all_calculated_xis.append(self.xi)

            self.xi = (1 - self.epsilon) * self.xi + np.matmul((self.epsilon * np.linalg.inv(self.zi)),
                                                               np.transpose(self.yi))

            self.gi_old = self.gi
            self.hi_old = self.hi

            self.hi = self.simulation_function_xtx_btx.get_hessian_fn(self.xi)
            self.gi = np.subtract(np.matmul(self.hi, self.xi),
                                  self.simulation_function_xtx_btx.get_gradient_fn(self.xi, self.function_constants))

            self.yi = self.yi + self.gi - self.gi_old
            self.zi = self.zi + self.hi - self.hi_old

            self.is_ready_to_transmit = True
            print(str(self.node_id) + "update_estimation - end")
            self.transmit_data()
        return

    def has_result_founded(self):
        """This method will check and verify if the calculated xi in this node has sufficiently converged. If for the
        last calculated xi, the difference between xi and x(i-1) is less than minimum accepted divergence that has
        been provided by the user, then the it would be considered that calculated xi has enough convergence to its
        target value. """
        self.is_convergence_sufficient = False
        if len(self.all_calculated_xis) > 10:
            self.is_convergence_sufficient = True
            for i in range(10):
                for j in range(self.all_calculated_xis[-(i + 1)].size):
                    if abs(abs(self.all_calculated_xis[-(i + 1)][j]) - abs(
                            self.all_calculated_xis[-(i + 1) - 1][j])) > self.minimum_accepted_divergence:
                        self.is_convergence_sufficient = False
        return self.is_convergence_sufficient
