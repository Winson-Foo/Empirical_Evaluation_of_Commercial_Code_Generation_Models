import videoflow.core.node as node

class FrameIndexSplitter(node.ProcessorNode):
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()

    def process(self, data):
        index, frame = data
        return frame