import string

metaslash = 1


def printNames():
    neal = 'neal'
    michelle = 'michele'
    eric = 5
    print("Local values: %(neal)S %(michele)s %(eric)" % locals())


class Nothing:
    def printValue(value):
        print(value)

    def set(self, value):
        self.value = value


def tryToDoSomething(self, value):
    try:
        import string
        if not value:
            raise (RuntimeError, "Hey, there's no value")
        printNames('a, b, c')
    except:
        traceback.print_exc()

def setGlobal(value=None):
    metaslash = value
    print('Old MetaSlash value is:', metaslash)
    useless = Nothing(5)
    print('a useless value is:', useless.valeu)


setGlobal(50)
