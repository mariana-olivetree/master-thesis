import spacy
import scispacy
import re
import pandas as pd
import pymupdf

#import sys
#sys.path.append('/data/gent/490/vsc49096')

"""
- Data loading and text extraction
- Text Cleaning
- Sentence tokenization
- Chunking
- Filtering and Noise reduction
"""

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


class PDFParser:
    def __init__(self, location):
        self.location = location

    def get_context_chunks(self,
                           num_sentence_chunk_size: int = 10,
                           min_token_length: int = 30) -> pd.DataFrame:
        doc = pymupdf.open(self.location) 
        pages_and_texts = [] 

        for page_number, page in enumerate(doc):
            text = page.get_text() #extraction of raw text from PDF from each page
            text = text_formatter(text) #Normalize whitespace/line breaks
            pages_and_texts.append({
                    "page_number": page_number,
                    "page_char_count": len(text), #includes whitesoaces between words
                    "page_word_count": len(text.split(" ")), 
                    "page_sentence_count_raw": len(text.split(". ")), #names of people are "." and sentences are ". "
                    "page_token_count": len(text) / 4,
                    "text": text
                })

        nlp = spacy.load("en_core_sci_sm")  #

        for item in pages_and_texts:
            doc = nlp(item["text"])
            item["sentences"] = [sent.text.strip() for sent in doc.sents] #sentence segmentation
            item["page_sentence_count_spacy"] = len(item["sentences"])

        for item in pages_and_texts:
            item["sentence_chunks"] = split_list(item["sentences"], num_sentence_chunk_size) #groups sentences into fixed size chunks
            item["num_chunks"] = len(item["sentence_chunks"])

        pages_and_chunks = []
        for item in pages_and_texts:
            for sentence_chunk in item["sentence_chunks"]:
                joined = " ".join(sentence_chunk).replace("  ", " ").strip() #joins the sentences of a chunks into a single string
                joined = re.sub(r'\.([A-Z])', r'. \1', joined)
                pages_and_chunks.append({
                        "page_number": item["page_number"],
                        "sentence_chunk": joined,
                        "chunk_char_count": len(joined),
                        "chunk_word_count": len(joined.split(" ")),
                        "chunk_token_count": len(joined) / 4 #approximate token count
                    })

        df = pd.DataFrame(pages_and_chunks)
        return pd.DataFrame(df[df["chunk_token_count"] > min_token_length].reset_index(drop=True))
