#from GRSlib.parallel_tools import ParallelTools
from GRSlib.converters.convert import Convert
from GRSlib.converters.sections.lammps_ace import Ace
from GRSlib.converters.sections.lammps_base import Base
#from GRSlib.converters.lammps_snap import Snap

def convert(converter_name, pt, cfg):
    """Converter Factory from (xyz) to (D)"""
    instance = search(converter_name)

    if cfg.args.verbose:
        pt.single_print("Using {} as Descriptors from".format(converter_name))

    instance.__init__(converter_name, pt, cfg)
    return instance


def search(converter_name):
    instance = None
    for cls in Convert.__subclasses__():
        if cls.__name__.lower() == converter_name.lower():
            instance = Convert.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in descriptor types".format(converter_name))
    else:
        return instance
