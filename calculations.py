# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

#import csv
import json
#import os

import datasets
from datasets import load_dataset
from datasets.tasks import QuestionAnsweringExtractive
load_dataset("https://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:x-oauth-basic@github.com/JorntHogema/calculations/blob/main/")

logger = datasets.logging.get_logger(__name__)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/JorntHogema/calculations"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "https://github.com/JorntHogema/calculations/blob/e010d348f0c4ac89e154198dbc7bc4d1026e8204/"
_URLS = {
    "train": _URL + "calculations.json",
    "dev": _URL + "calculations-dev.json"
}


class CalculationsConfig(datasets.BuilderConfig):
    """BuilderConfig Calculations"""

    def __init__(self, **kwargs):
        """BuilderCOnfig for Calulations
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CalculationsConfig, self).__init__(**kwargs)

class Calculations(datasets.GeneratorBasedBuilder):
    """Calculations: the calculation answers dataset"""

    BUILDER_CONFIGS = [
        CalculationsConfig(
        name="plain_text",
        version=datasets.version("1.0.0",""),
        description="plain_text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            ),
        supervised_keys=None,
        homepage=_HOMEPAGE,
        # License for the dataset if available
        license=_LICENSE,
        # Citation for the dataset
        citation=_CITATION,
        task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],        
        ),
    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]
        
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not importnt in itself, but must be unique for each example.
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.loads(f)
            for article in data["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        answers = [answers["text"] for answers in qa["answers"]]
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": "" if split == "test" else data["answers"],
                        }
                        key += 1
