def print_versions(libs = []):
    """ Prints Python version and name version pairs for specified libs """
    
    import sys
    
    print('Python', sys.version)

    for lib in libs:
        print(' - ', lib.__name__, lib.__version__)
        

def unpickle(path):
    """ Unpickles .pkl file in restriced mode """
    
    from pickle import Unpickler, UnpicklingError
    
    class RestrictedUnpickler(Unpickler):
        def find_class(self, module, name):
            raise UnpicklingError('{} in {} is restricted'.format(name, module))
            
    with open(path, mode='r+b') as binary:
        return RestrictedUnpickler(binary).load()
    