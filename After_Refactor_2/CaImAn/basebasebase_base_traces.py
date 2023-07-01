from trace import Trace

if __name__ == "__main__":
    tracedata = Trace(3 + np.random.random((2000, 4)), fr=30, start_time=0)
    tracedata_dff = tracedata.compute_dff()