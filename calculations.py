import json
import datasets

_URL = "https://github.com/JorntHogema/calculations/blob/e010d348f0c4ac89e154198dbc7bc4d1026e8204/"
_URLS = {"train": _URL + "calculations.json"}
    
def _info(self):
    return datasets.DatasetInfo(
        description=_DESCRIPTION,
        features=datasets.Features(
            {
                "id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answers": datasets.Value("string"),
            }
        ),
        # No default supervised_keys (as we have to pass both question
        # and context as input).
        supervised_keys=None,
        homepage="https://rajpurkar.github.io/SQuAD-explorer/",
        citation=_CITATION,
    )
