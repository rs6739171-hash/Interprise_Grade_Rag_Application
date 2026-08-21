from bs4 import BeautifulSoup
import logfire

def parse_html(file_path: str):
    """Parses HTML content using BeautifulSoup. Cleans scripts, styles, and extracts readable text for RAG."""

    with logfire.span("HTML Parsing", file=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')

            # Remove junk: scripts, styles, metadata
            # NOTE: real BeautifulSoup/HTML tag names are singular —
            # "style" and "noscript", not "styles"/"noscripts".
            for tag in soup(["style", "meta", "script", "noscript"]):
                tag.decompose()

            # 2. Extract Text
            text = soup.get_text(separator="\n")

            # 3. Clean Whitespace (collapse multiple newlines)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)

            return text_clean
        except Exception as e:
            logfire.error(f"HTML Parse Failed: {e}")
            raise e