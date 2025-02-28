from GRSlib.parallel_tools import ParallelTools
from GRSlib.converters.convert import Convert
from GRSlib.converters.lammps_pace import LammpsPace

def convert(converter_name, pt, cfg):
    """Converter Factory from (xyz) to (D)"""
    instance = search(converter_name)
    #pt = ParallelTools()
    if cfg.args.verbose:
        pt.single_print("Using {} as Descriptors from".format(converter_name))

    instance.__init__(converter_name, pt, cfg)
    return instance


def search(converter_name):
    instance = None

    # loop over subclasses 

    for cls in Convert.__subclasses__():

        # loop over sublcasses of this subclass (e.g. LammpsBase has LammpsSnap and LammpsPace)

        for cls2 in cls.__subclasses__():
            if cls2.__name__.lower() == converter_name.lower():
                instance = Convert.__new__(cls2)

    if instance is None:
        raise IndexError("{} was not found in descriptor sets available to LAMMPS".format(converter_name))
    else:
        return instance

