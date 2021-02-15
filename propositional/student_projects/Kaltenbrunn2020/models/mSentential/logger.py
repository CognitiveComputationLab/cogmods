"""very very very simple logging by Guerth:
https://github.com/CognitiveComputationLab/cogmods/blob/master/modal/student_projects/2019_guerth/models/mmodalsentential/logger.py
"""

logger_enabled = True


def logging(*args, level="INFO", **kwargs):
    """If enabled, log/print message with tags, e.g. INFO:message

    Keyword Arguments:
        level {str} -- the tag before the message (default: {"INFO"})
    """
    if logger_enabled:
        print(level + ": " + " ".join(map(str, args)), **kwargs)
        # if level == "MODEL":
        #     print(" ".join(map(str, args)), **kwargs)
        # else:
        #     print(level + ": " + " ".join(map(str, args)), **kwargs)
