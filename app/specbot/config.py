import json

class Config(object):
    def __init__(self, conf_file="config.json"):
        with open(conf_file, "rt", encoding="utf8") as f :
            conf = json.load(f)

        self.chunk_size : str = conf['chunk_size']
        self.chunk_overlap : str = conf['chunk_overlap']

        