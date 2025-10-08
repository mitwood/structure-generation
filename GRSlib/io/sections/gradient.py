from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Gradient(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['soft_strength', 'ml_strength', 'nsteps', 'temperature','min_type','randomize_comps']
        self._check_section()
        self.soft_strength = self.get_value("GRADIENT", "soft_strength", 1.0)
        self.ml_strength = self.get_value("GRADIENT", "ml_strength", 1.0)
        self.nsteps = self.get_value("GRADIENT", "nsteps", 100)
        self.temperature = self.get_value("GRADIENT", "temperature", 10.0)
        self.min_type = self.get_value("GRADIENT", "min_type", 'line')
        self.rand_comp = self.get_value("GRADIENT", "randomize_comps", "False", interpreter="bool")
        self.delete()
