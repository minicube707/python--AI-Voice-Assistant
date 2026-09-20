from dataclasses import dataclass, asdict, field
import json


@dataclass
class Command:
    intent: str = "unknown"
    argument: dict = field(default_factory=dict)
    text: str = ""
    status: str = "ok"
    

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


    def to_dict(self) -> dict:
        return asdict(self)


