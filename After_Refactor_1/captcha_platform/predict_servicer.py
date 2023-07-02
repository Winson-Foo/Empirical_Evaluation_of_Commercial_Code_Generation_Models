import grpc

from compat import grpc_pb2 as grpc__pb2


class PredictServicer(object):
  """Server-side implementation of Predict service."""

  def predict(self, request, context):
    """Handles the predict RPC.

    Args:
      request: A grpc__pb2.PredictRequest.
      context: A grpc.ServicerContext.

    Raises:
      grpc.RpcError: If the method is not implemented.

    Returns:
      The response for the predict RPC.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, 'Method not implemented!')