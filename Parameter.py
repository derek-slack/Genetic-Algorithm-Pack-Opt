class Parameter:
    def __init__(self, name, type, value, n_bit, max_bit, min_bit):
        self.name = name
        self.type = type
        self.value = value
        self.n_bit = n_bit

        self.min_bit = min_bit
        self.max_bit = max_bit

