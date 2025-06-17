from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Target(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['target_fname', 'start_fname']
        self._check_section()
        self.target_fname = self.get_value("TARGET", "target_fname", None)
        self.start_fname = self.get_value("TARGET", "start_fname", None)
        self.delete()
