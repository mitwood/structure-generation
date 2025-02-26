from io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Output(Target):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['avg_weight', 'var_weight']
        self._check_section()
        self.target = self.get_value("TARGET", "target_fname", None)
        self.target = self.get_value("TARGET", "start_fname", None)
        self.delete()
