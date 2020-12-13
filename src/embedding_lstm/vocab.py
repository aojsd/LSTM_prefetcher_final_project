
class Vocab:
    def __init__(self, all_keys=None):
        self.key_to_val = {}
        self.val_to_key = {}
        self.counter = 0

        if all_keys is not None:
            for key in all_keys:
                self.add_key(key)

    def __len__(self):
        return self.counter
        
    def get_val(self, key):
        # Return -1 by default to treat pruned out deltas as unknown
        # Assumption: all the keys remain the same, so we can use
        # `self.counter` as a dummy value (kind of like [unk] in 
        # neural translation models).
        return self.key_to_val.get(key, self.counter)

    def get_key(self, val):
        return self.val_to_key.get(val, None)

    def add_key(self, key):
        if key not in self.key_to_val:
            self.key_to_val[key] = self.counter
            self.val_to_key[self.counter] = key
            self.counter += 1