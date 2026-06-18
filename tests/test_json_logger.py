import json
from neurooptai.utils.json_logger import JSONLogger


def test_json_logger_writes_event(tmp_path):
    log_path = tmp_path / "test_log.jsonl"
    logger = JSONLogger(str(log_path))

    record = logger.log({
        "epoch": 1,
        "decision": {"action": "continue_training"}
    })

    assert log_path.exists()
    assert record["epoch"] == 1
    assert "timestamp" in record

    line = log_path.read_text().strip()
    parsed = json.loads(line)

    assert parsed["epoch"] == 1
    assert parsed["decision"]["action"] == "continue_training"
