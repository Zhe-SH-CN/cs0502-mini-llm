from . import courseqa, jsonout, intent

BUILDERS = {"courseqa": courseqa.build, "jsonout": jsonout.build, "intent": intent.build}

__all__ = ["BUILDERS", "courseqa", "jsonout", "intent"]
