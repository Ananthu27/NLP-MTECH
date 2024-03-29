def stem(word):
    suffixes = {
        1: [("", "s", "es")],
        2: [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")],
        3: [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("bli", "ble"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ],
        4: [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ],
    }

    for length in range(len(word), 0, -1):
        if length > 2:
            for old, new in suffixes[length]:
                if word.endswith(old):
                    return word[:-len(old)] + new
        else:
            return word