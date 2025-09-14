# ACTMS - Anti-Corruption Tender Management System

## Overview

ACTMS (Anti-Corruption Tender Management System) is a comprehensive web application designed to ensure transparency and integrity in government tender processes. The system combines traditional tender management with AI-powered anomaly detection to identify suspicious bidding patterns and maintain accountability. The platform serves as a centralized hub where government departments can publish tenders, companies can submit bids, and administrators can monitor the entire process for irregularities using machine learning algorithms and comprehensive audit logging.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
The system uses a **Flask-based REST API** architecture with modular service components. The main application (`app.py`) serves as the central controller that orchestrates various specialized services:

- **Database Service**: SQLite-based data persistence layer handling tenders, bids, and audit logs
- **ML Service**: Isolation Forest-based anomaly detection system for identifying suspicious bid patterns
- **Chatbot Service**: Gemini AI integration with FAQ fallback for user assistance
- **NLP Service**: Text analysis service using spaCy for proposal evaluation and quality assessment
- **File Handler**: Secure file upload and validation system with hash verification

The Flask application is configured for both development and production environments with appropriate CORS settings and security measures.

### Frontend Architecture
The system uses a **static HTML/CSS/JavaScript frontend** with Tailwind CSS for styling and Chart.js for data visualization. The frontend communicates with the backend through RESTful API calls using fetch() for all data operations. The interface features a responsive sidebar navigation, dashboard with statistics, and dedicated sections for tender management, bid submission, and system monitoring.

### Data Storage Solutions
**SQLite Database** serves as the primary data store with three main tables:
- **Tenders**: Store tender information including metadata, deadlines, and file references
- **Bids**: Store bid submissions with anomaly scores and suspicious flags
- **Audit Logs**: Track all system actions for transparency and accountability

File storage uses a **local filesystem approach** with uploads stored in `/uploads` directory and trained ML models persisted in `/models` directory using joblib.

### Machine Learning Architecture
The system employs **Isolation Forest algorithm** from scikit-learn for unsupervised anomaly detection with:
- 10% contamination threshold for flagging suspicious bids
- Feature extraction from bid amounts, proposal text quality, timing patterns, and company information
- Model persistence and retraining capabilities
- Performance metrics tracking for model evaluation

Features analyzed include bid amounts, proposal length, company name patterns, submission timing, NLP-derived text quality metrics, and readability scores.

### Security and Validation
Multi-layered security approach includes:
- **File validation**: Size limits (15MB), extension filtering (PDF, DOC, DOCX, TXT, RTF), and content verification
- **Hash verification**: SHA-256 hashing for file integrity checking
- **Audit logging**: Comprehensive action tracking for all major operations
- **Input sanitization**: Secure filename handling and request validation

### API Design
RESTful API structure with JSON responses covering:
- Dashboard statistics (`/api/dashboard`)
- Tender management (`/api/tenders`)
- Bid submission and retrieval (`/api/bids`)
- Anomaly detection (`/api/bids/suspicious`)
- Alert management (`/api/alerts`)
- ML model operations (`/api/model/train`, `/api/model/test`)
- Chatbot interaction (`/api/chat`)

## External Dependencies

### AI and Machine Learning
- **Google Gemini API**: Powers the intelligent chatbot service for user assistance and FAQ responses
- **scikit-learn**: Provides Isolation Forest algorithm for anomaly detection in bid submissions
- **spaCy**: Handles natural language processing for text analysis and proposal quality assessment

### Frontend Libraries
- **Tailwind CSS**: Utility-first CSS framework for responsive UI design
- **Chart.js**: JavaScript library for data visualization and dashboard charts

### Python Framework and Libraries
- **Flask**: Core web framework with CORS support for API development
- **SQLite**: Embedded database for data persistence
- **joblib**: Model serialization for ML persistence
- **pandas/numpy**: Data manipulation and numerical operations

### File Processing
- **Werkzeug**: Secure file handling utilities for upload validation
- **hashlib**: SHA-256 hashing for file integrity verification

### Development Tools
- Local file system for uploads and model storage
- JSON-based FAQ system for chatbot fallback responses
- Environment-based configuration for production deployment