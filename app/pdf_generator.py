import os
from pathlib import Path

import markdown
import pdfkit


def get_wkhtmltopdf_config():
    """
    Configure wkhtmltopdf path.
    - On Windows, set WKHTMLTOPDF_CMD env var if needed.
    - Example:
      setx WKHTMLTOPDF_CMD "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    """

    wkhtml_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"

    if wkhtml_path:
        return pdfkit.configuration(wkhtmltopdf=wkhtml_path)
    
    # On Linux/Mac, wkhtmltopdf often works without explicit path
    return None


def markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """
    Convert a Markdown file to PDF using wkhtmltopdf via pdfkit.
    """
    md_path = Path(md_path)

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code"])

    config = get_wkhtmltopdf_config()
    
    # If config is None, pdfkit will try system default
    pdfkit.from_string(html, str(pdf_path), configuration=config)