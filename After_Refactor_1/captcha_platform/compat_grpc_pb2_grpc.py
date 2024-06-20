import grpc

from compat import grpc_pb2 as grpc__pb2
from predict_servicer import PredictServicer


def add_PredictServicer_to_server(servicer, server):
  """Adds the PredictServicer to the given server.

  Args:
    servicer: A PredictServicer instance.
    server: A grpc.Server instance.
  """
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