from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='grpc.proto',
    package='',
    syntax='proto3',
    serialized_options=None,
    serialized_pb='...'
)


_PREDICTREQUEST = _descriptor.Descriptor(
    name='PredictRequest',
    full_name='PredictRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='image',
            full_name='PredictRequest.image',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            # ...
        ),
        # ... other fields
    ],
    # ...
)


_PREDICTRESULT = _descriptor.Descriptor(
    name='PredictResult',
    full_name='PredictResult',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='result',
            full_name='PredictResult.result',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            # ...
        ),
        # ... other fields
    ],
    # ...
)


DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictResult'] = _PREDICTRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


class PredictRequest(_message.Message):
    DESCRIPTOR = _PREDICTREQUEST
    pass


_sym_db.RegisterMessage(PredictRequest)


class PredictResult(_message.Message):
    DESCRIPTOR = _PREDICTRESULT
    pass


_sym_db.RegisterMessage(PredictResult)


_PREDICT = _descriptor.ServiceDescriptor(
    name='Predict',
    full_name='Predict',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    serialized_start=212,
    serialized_end=267,
    methods=[
        _descriptor.MethodDescriptor(
            name='predict',
            full_name='Predict.predict',
            index=0,
            containing_service=None,
            input_type=_PREDICTREQUEST,
            output_type=_PREDICTRESULT,
            serialized_options=None,
        ),
    ],
)

_sym_db.RegisterServiceDescriptor(_PREDICT)