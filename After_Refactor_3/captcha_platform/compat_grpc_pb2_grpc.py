import grpc
from compat import grpc_pb2 as grpc__pb2

class PredictStub(object):
    """Client-side stub for the Predict service."""
    
    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.predict = channel.unary_unary(
            '/Predict/predict',
            request_serializer=grpc__pb2.PredictRequest.SerializeToString,
            response_deserializer=grpc__pb2.PredictResult.FromString,
        )

class PredictServicer(object):
    """Server-side implementation of the Predict service."""
    
    def predict(self, request, context):
        """Handler for the predict method."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PredictServicer_to_server(servicer, server):
    """Adds the Predict servicer to the server."""
    rpc_method_handlers = {
        'predict': grpc.unary_unary_rpc_method_handler(
            servicer.predict,
            request_deserializer=grpc__pb2.PredictRequest.FromString,
            response_serializer=grpc__pb2.PredictResult.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'Predict', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))