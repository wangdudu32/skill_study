"""一个字符对应一个 token，没见过的字符用 0 表示。"""


class CharTokenizer:
    unk_id = 0

    def __init__(self, characters: list[str]):
        if not characters or any(not isinstance(char, str) or len(char) != 1 for char in characters):
            raise ValueError("词表必须包含至少一个字符，每个条目必须是单个 Unicode 字符")
        if len(set(characters)) != len(characters):
            raise ValueError("词表不能包含重复字符")
        self.characters = tuple(characters)
        self.char_to_id = {char: index + 1 for index, char in enumerate(self.characters)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(char, self.unk_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token id 超出词表范围：{token_id}")
            pieces.append("[UNK]" if token_id == self.unk_id else self.characters[token_id - 1])
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {"type": "char", "characters": list(self.characters)}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        if data.get("type") != "char":
            raise ValueError("不支持的 tokenizer 类型")
        return cls(data["characters"])
