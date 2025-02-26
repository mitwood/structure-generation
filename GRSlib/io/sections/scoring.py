from io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Scoring(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['moments', 'moments_coeff', 'moment_bonus', 'moments_cross_coeff','moment_cross_bonus','attractor_target','exact_distribution']
        self._check_section()
        self.scoring = self.get_value("SCORING", "moments", [1,2], interpreter=int)
        self.scoring = self.get_value("SCORING", "moments_coeff", [1.0, 0.1])
        self.scoring = self.get_value("SCORING", "moment_bonus", [1.0, 5.0])
        self.scoring = self.get_value("SCORING", "moments_cross_coeff", [1.0, 0.1])
        self.scoring = self.get_value("SCORING", "moment_cross_bonus", [1.0, 5.0])
        self.scoring = self.get_value("SCORING", "attractor_target", True, interpreter=bool)
        self.scoring = self.get_value("SCORING", "exact_distribution", False, interpreter=bool)
        self.delete()

