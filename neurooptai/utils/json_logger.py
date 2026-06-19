import json
from datetime import datetime, UTC


class JSONLogger:
    """
    Guarda esdeveniments de NeuroOptAI en format JSON Lines.
    Cada línia és una decisió o estat del sistema.
    """

    def __init__(self, path="neurooptai_log.jsonl"):
        self.path = path

    def log(self, event):
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            **event,
        }

        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record
