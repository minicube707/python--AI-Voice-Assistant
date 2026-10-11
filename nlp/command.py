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

     
    def __repr__(self):

        string = f"{self.intent=}\n"
        string += "self.argument="
        string += json.dumps(self.argument, indent=4, ensure_ascii=False)
        string += "\n"
        string += f"{self.text=}\n"
        string += f"{self.status=}\n"

        return string