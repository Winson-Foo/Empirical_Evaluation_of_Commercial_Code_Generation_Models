import grpc

from compat import grpc_pb2 as grpc__pb2


class PredictStub(object):
  """Client-side stub for Predict service."""

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