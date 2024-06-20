import multiprocessing

bind = '0.0.0.0:19951'
workers = multiprocessing.cpu_count() * 2 + 1
backlog = 2048
# worker_class = "gevent"
debug = False
daemon = True
proc_name = 'gunicorn.pid'
pidfile = 'debug.log'
errorlog = 'error.log'
accesslog = 'access.log'
loglevel = 'info'
timeout = 10