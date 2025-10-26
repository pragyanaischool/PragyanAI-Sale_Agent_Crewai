import pypdf  # Changed from PyPDF2
import docx
from typing import Type
from pydantic import BaseModel, Field

# Corrected import: BaseTool is now in 'crewai'
from crewai import BaseTool

# --- 1. Define Input Schema ---
# This tells the agent *how* to use the tool.
class DocumentReaderInput(BaseModel):
    file_path: str = Field(description="The full path to the .pdf or .docx file.")

# --- 2. The Tool Logic ---

class DocumentReaderTool(BaseTool):
    name: str = "DocumentReaderTool"
    description: str = "Reads text content from a specified .pdf or .docx file."
    args_schema: Type[BaseModel] = DocumentReaderInput

    def _run(self, file_path: str) -> str:
        """
        Reads text from the given file path based on its extension.
        """
        text = ""
        try:
            if file_path.endswith(".pdf"):
                # Handle PDF files
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)  # Changed from PyPDF2.PdfReader
                    for page in reader.pages:
                        text += page.extract_text() or ""
                if not text:
                    return f"Error: Could not extract text from PDF: {file_path}. The file might be image-based or corrupted."
                        
            elif file_path.endswith(".docx"):
                # Handle DOCX files
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
                if not text:
                    return f"Error: No text found in DOCX file: {file_path}."

            else:
                return f"Error: Unsupported file format: {file_path}. Only .pdf and .docx are supported."

            return text
        
        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"

