class SignUtils:
    @staticmethod
    def timestamp():
        return int(time.time())

    @staticmethod
    def md5(string):
        return hashlib.md5(string.encode("utf-8")).hexdigest()