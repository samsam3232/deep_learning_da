import dalib.vision.datasets as ds

DATASETS_DICT = {"Office31" : ds.office31, "OfficeHome" : ds.officehome, "OfficeCaltech" : ds.officecaltech,
                 "DomainNet" : ds.domainnet, "VisDA2017" : ds.visda2017}
DATASTS_DOMAINS = {"Office31":["A", "W", "D"], "OfficeHome" : ["Ar", "Cl", "Pr", "Rw"], "OfficeCaltech" : ["A", "W", "D",
                   "C"], "DomainNet" : ["c", "i", "s", "r", "p", "q"], "VisDA2017" : ["T", "V"] }

def get_loaders():

    return None