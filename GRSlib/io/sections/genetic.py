from io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Genetic(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['soft_strength', 'ml_strength', 'nsteps', 'temperature','min_type','randomize_comps']
        self._check_section()
        self.genetic = self.get_value("MOTION", "soft_strength", 1.0)
        self.genetic = self.get_value("MOTION", "ml_strength", 1.0)
        self.genetic = self.get_value("MOTION", "nsteps", 100)
        self.genetic = self.get_value("MOTION", "temperature", 0.0)
        self.genetic = self.get_value("MOTION", "min_type", 'line')
        self.genetic = self.get_value("MOTION", "randomize_comps", "False", interpreter="bool")
        self.delete()
