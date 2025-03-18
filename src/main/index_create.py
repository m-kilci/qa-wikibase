import elasticsearch
from elasticsearch import helpers
import json

es = elasticsearch.Elasticsearch("http://localhost:9200")
if not es.indices.exists(index="wikibase_index"):
    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 2
    }
    mappings = {
        "properties": {
            "doc_id": {"type": "integer", "index": "false"},
            "title": {"type": "text", "index": "true"},
            "text": {"type": "text", "index": "true"}
        }
    }
    es.indices.create(index="wikibase_index",
                      settings=settings,
                      mappings=mappings)

es.indices.flush()  # persist changes (memory -> disk)
es.indices.refresh()  # makes changes available for search

file_path = "wikibase.jsonl"


def read_documents_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            yield {
                "_index": "wikibase_index",
                "_id": doc["doc_id"],
                "_source": {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": doc["text"]
                }
            }


helpers.bulk(es, read_documents_from_file(file_path))
