from camel_tools.utils.charmap import CharMapper
import arabic_reshaper
from .letters import LETTERS_ARABIC
# from .ligatures import LETTERS_LIGATURES
import numpy as np

# compile a dictionary mapping each letter to its possible allographic variants
# for each value in LETTERS_ARABIC (a list), take each non-empty string as a key, and the dropnaped list as the value
allographic_map = {}
for a_form, contextual_forms in LETTERS_ARABIC.items():
    variants = [v for v in contextual_forms if v]  # drop empty strings
    for char in variants + [a_form]:
        allographic_map[char] = variants

LIGATURES = (    ('ARABIC LIGATURE LAM WITH ALEF', (
        '\u0644\u0627', ('\uFEFB', '', '', '\uFEFC'),
    )),
    ('ARABIC LIGATURE LAM WITH ALEF MAKSURA', (
        '\u0644\u0649', ('\uFC43', '', '', '\uFC86'),
    )),
    ('ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE', (
        '\u0644\u0623', ('\uFEF7', '', '', '\uFEF8'),
    )),
    ('ARABIC LIGATURE LAM WITH ALEF WITH HAMZA BELOW', (
        '\u0644\u0625', ('\uFEF9', '', '', '\uFEFA'),
    )),
    ('ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE', (
        '\u0644\u0622', ('\uFEF5', '', '', '\uFEF6'),
    )))
# for liga_name, (a_forms, contextual_forms) in LETTERS_LIGATURES.items():
    


def reshape_arabic_text(text: str) -> str:
    """
    Reshape Arabic text for proper rendering.

    Args:
        text (str): The input Arabic text.
    Returns:
        str: The reshaped Arabic text.
    """
    reshaped_text = arabic_reshaper.reshape(text)
    return reshaped_text

def HSB_transliterate(text: str) -> str:
    """
    Transliterate Arabic text using the HSB scheme.

    Args:
        text (str): The input Arabic text.
    Returns:
        str: The transliterated text.
    """
    transliterator = CharMapper.builtin_mapper('ar2hsb')
    transliterated_text = transliterator(text)
    return transliterated_text

def get_allographic_variant(word: str, probability: float = 0.1) -> str:
    """
    Get an allographic variant of the given Arabic word with a certain probability.

    Args:
        word (str): The input Arabic word.
        probability (float): The probability of changing a letter in the word to one of its allographic variants.
    Returns:
        str: The original or allographic variant of the word.
    """
    new_word = ''
    for char in word:
        # if char not in allographic_map:
            # print(f"Character '{char}' not found in allographic map.")
        n = np.random.rand()
        if char in allographic_map and n < probability:
            variants = allographic_map[char]
            new_char = np.random.choice(variants)
            new_word += new_char
        else:
            new_word += char

    return new_word