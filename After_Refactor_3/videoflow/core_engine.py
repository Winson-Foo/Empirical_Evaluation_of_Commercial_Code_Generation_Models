class MessageSender:
    '''Sends output message to a place where the child task will receive it.'''
    def send_message(self, message):
        raise NotImplementedError('MessageSender subclass must implement method')

class MessageReceiver:
    '''Receives input message from a place where the parent task sent it.'''
    def receive_message(self):
        raise NotImplementedError('MessageReceiver subclass must implement method')

class TerminationMessageSender:
    '''Sends termination message to all the nodes in the graph.'''
    def send_termination_message(self, message):
        raise NotImplementedError('TerminationMessageSender subclass must implement method')

class TerminationMessageReceiver:
    '''Receives termination message from the topmost node.'''
    def receive_termination_message(self):
        raise NotImplementedError('TerminationMessageReceiver subclass must implement method')

from abc import ABC, abstractmethod

class BaseExecutionEngine(ABC):
    '''Defines the interface of the `execution environment`.'''
    @abstractmethod
    def create_and_start_processes(self, tasks_data):
        '''
        Create and start processes for all the tasks in the graph.

        Arguments:
        - tasks_data: list of tuples. The list is of the form [(node : Node, node_index : int, parent_index : int, has_children : bool)]
        '''
        pass

    @abstractmethod
    def signal_flow_termination(self):
        '''Signals the execution environment that the flow needs to stop.'''
        pass

    @abstractmethod
    def join_task_processes(self):
        '''
        Makes the calling process sleep until all task processes have finished processing.
        '''
        pass

class LocalExecutionEngine(BaseExecutionEngine):
    '''Implementation of the BaseExecutionEngine for local execution.'''
    def create_and_start_processes(self, tasks_data):
        '''Create and start processes using the multiprocessing library.'''
        # implementation details

    def signal_flow_termination(self):
        '''Signal the flow termination using a queue or another multiprocessing tool.'''
        # implementation details

    def join_task_processes(self):
        '''Join all the processes using the multiprocessing library.'''
        # implementation details

class BaseGraphElement:
    '''Defines the interface of a graph element.'''
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compute(self, inputs):
        pass

class Node(BaseGraphElement):
    '''Defines a node in the graph.'''
    def __init__(self, name, function):
        super().__init__(name)
        self.function = function

    def compute(self, inputs):
        return self.function(*inputs)

class Task(BaseGraphElement):
    '''Defines a task in the graph.'''
    def __init__(self, name, function):
        super().__init__(name)
        self.function = function

    def compute(self, inputs):
        result = self.function(*inputs)
        if not isinstance(result, tuple):
            result = (result,)
        return result