import os
import hashlib
from typing import Tuple, Optional
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

class FileHandler:
    def __init__(self, upload_folder: str = 'uploads'):
        self.upload_folder = upload_folder
        self.allowed_extensions = {'pdf', 'doc', 'docx', 'txt', 'rtf'}
        self.max_file_size = 15 * 1024 * 1024  # 15MB in bytes
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def validate_file(self, file: FileStorage) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if not file or not file.filename or file.filename == '':
            return False, "No file selected"
        
        if not self.allowed_file(file.filename):
            return False, f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
        
        # Check file size by reading content
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size:
            return False, f"File size exceeds maximum limit of {self.max_file_size // (1024*1024)}MB"
        
        return True, "File is valid"
    
    def save_file(self, file: FileStorage, prefix: str = '') -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Save uploaded file and return file path, hash, and any error message
        Returns: (file_path, file_hash, error_message)
        """
        try:
            # Validate file
            is_valid, message = self.validate_file(file)
            if not is_valid:
                return None, None, message
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset for potential re-reading
            
            # Generate file hash
            file_hash = self.get_file_hash(file_content)
            
            # Generate secure filename
            if not file.filename:
                return None, None, "Invalid filename"
            original_filename = secure_filename(file.filename)
            if prefix:
                filename = f"{prefix}_{file_hash[:8]}_{original_filename}"
            else:
                filename = f"{file_hash[:8]}_{original_filename}"
            
            file_path = os.path.join(self.upload_folder, filename)
            
            # Check if file with same hash already exists
            if os.path.exists(file_path):
                return file_path, file_hash, None
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            return file_path, file_hash, None
            
        except Exception as e:
            return None, None, f"Error saving file: {str(e)}"
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file safely"""
        try:
            if os.path.exists(file_path) and file_path.startswith(self.upload_folder):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False
    
    def get_file_info(self, file_path: str) -> Optional[dict]:
        """Get file information"""
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                return {
                    'size': stat.st_size,
                    'created': stat.st_ctime,
                    'modified': stat.st_mtime,
                    'filename': os.path.basename(file_path)
                }
            return None
        except Exception:
            return None
    
    def read_text_file(self, file_path: str) -> Optional[str]:
        """Read text content from supported file types"""
        try:
            if not os.path.exists(file_path):
                return None
            
            filename = file_path.lower()
            
            if filename.endswith('.txt') or filename.endswith('.rtf'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif filename.endswith('.pdf'):
                # For PDF files, we'd need PyPDF2 or similar
                # For now, return a placeholder
                return "[PDF content - requires PDF processing library]"
            elif filename.endswith(('.doc', '.docx')):
                # For DOC/DOCX files, we'd need python-docx or similar
                # For now, return a placeholder
                return "[DOC/DOCX content - requires document processing library]"
            
            return None
        except Exception:
            return None