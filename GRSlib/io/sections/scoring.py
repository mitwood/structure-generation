from io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Scoring(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['avg_weight', 'var_weight', 'ensemble_weight', 'self_weight','avg_bonus','avg_ensemble_bonus','var_bonus','var_ensemble_bonus']
        self._check_section()
        self.scoring = self.get_value("SCORING", "avg_weight", 1.0)
        self.scoring = self.get_value("SCORING", "var_weight", 0.0)
        self.scoring = self.get_value("SCORING", "ensemble_weight", 0.0)
        self.scoring = self.get_value("SCORING", "self_weight", 0.5)
        self.scoring = self.get_value("SCORING", "avg_bonus", 0.0)
        self.scoring = self.get_value("SCORING", "avg_ensemble_bonus", 0.0)
        self.scoring = self.get_value("SCORING", "var_bonus", 0.0)
        self.scoring = self.get_value("SCORING", "var_ensemble_bonus", 0.0)
        self.delete()

