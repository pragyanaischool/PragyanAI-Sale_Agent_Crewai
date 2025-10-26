import os
from crewai_tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Import readers
from pypdf import PdfReader
from docx import Document

class ReadDocumentInput(BaseModel):
    """Input model for DocumentReaderTool."""
    file_path: str = Field(..., description="The local file path to the document to be read.")

class DocumentReaderTool(BaseTool):
    """
    A CrewAI tool to read text content from various document types (PDF, DOCX, TXT).
    
    This tool takes a file path, determines the file type, and uses the
    appropriate library to extract and return its text content.
    """
    name: str = "Document Reader Tool"
    description: str = "Reads a document from a specified file path and returns its text content."
    args_schema: Type[BaseModel] = ReadDocumentInput
    
    def _run(self, file_path: str) -> str:
        """
        The main execution method for the tool.
        
        Args:
            file_path: The local path to the document.
            
        Returns:
            The extracted text content of the document as a string.
            Returns an error message if the file type is unsupported or reading fails.
        """
        if not os.path.exists(file_path):
            return f"Error: File not found at path: {file_path}"
            
        # Get the file extension and determine the file type
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            if file_extension == '.pdf':
                return self._read_pdf(file_path)
            elif file_extension == '.docx':
                return self._read_docx(file_path)
            elif file_extension == '.txt':
                return self._read_txt(file_path)
            else:
                return f"Error: Unsupported file type: {file_extension}. Only .pdf, .docx, and .txt are supported."
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    def _read_pdf(self, file_path: str) -> str:
        """Extracts text from a PDF file."""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")

    def _read_docx(self, file_path: str) -> str:
        """Extracts text from a DOCX file."""
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")

    def _read_txt(self, file_path: str) -> str:
        """Extracts text from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read TXT: {str(e)}")

# Example of how to instantiate the tool
# if __name__ == "__main__":
#     # This part is for testing the tool directly
#     # Create a dummy test.txt file
#     with open("test.txt", "w") as f:
#         f.write("This is a test text file.")
        
#     tool = DocumentReaderTool()
#     content = tool.run(file_path="test.txt")
#     print("--- Reading TXT ---")
#     print(content)
    
#     # You would need to have a real PDF and DOCX file to test the others
#     # content_pdf = tool.run(file_path="path/to/your.pdf")
#     # print("\n--- Reading PDF ---")
#     # print(content_pdf)
