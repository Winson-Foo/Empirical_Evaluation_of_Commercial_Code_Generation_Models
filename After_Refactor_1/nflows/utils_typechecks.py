class TypeChecker:

    @staticmethod
    def is_bool(value):
        return isinstance(value, bool)

    @staticmethod
    def is_positive_int(value):
        return isinstance(value, int) and value > 0

    @staticmethod
    def is_nonnegative_int(value):
        return isinstance(value, int) and value >= 0

    @staticmethod
    def is_power_of_two(value):
        return isinstance(value, int) and value > 0 and (value & (value - 1)) == 0