from watchdog.observers import Observer
from file_event_handler import FileEventHandler

if __name__ == "__main__":
    observer = Observer()
    file_event_handler = FileEventHandler()
    observer.schedule(file_event_handler, file_event_handler.model_conf_path, True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()