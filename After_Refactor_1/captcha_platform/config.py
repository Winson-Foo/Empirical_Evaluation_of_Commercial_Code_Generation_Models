class Config:
    bind = '0.0.0.0:19951'
    backlog = 2048
    debug = False
    daemon = True
    proc_name = 'gunicorn.pid'
    pidfile = 'debug.log'
    errorlog = 'error.log'
    accesslog = 'access.log'
    loglevel = 'info'
    timeout = 10

    @staticmethod
    def get_num_workers():
        return (multiprocessing.cpu_count() * 2) + 1