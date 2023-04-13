import traceback

METASLASH = 1


def print_names():
    neal = 'neal'
    michelle = 'michele'
    eric = 5
    print(f"Local values: {neal} {michelle} {eric}")


class Nothing:
    def __init__(self, value):
        self.value = value

    def print_value(self):
        print(self.value)


def try_do_something(value):
    try:
        if not value:
            raise RuntimeError("Hey, there's no value")
        print_names()
    except Exception as e:
        traceback.print_exc()


def set_global(value=None):
    global METASLASH
    METASLASH = value
    print('Old MetaSlash value is:', METASLASH)
    useless = Nothing(5)
    print('a useless value is:', useless.value)


set_global(50)
