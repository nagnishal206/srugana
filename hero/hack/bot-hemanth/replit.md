# ACTMS - Anti-Corruption Tender Management System

## Overview

ACTMS (Anti-Corruption Tender Management System) is a comprehensive web application designed to ensure transparency and integrity in government tender processes. The system combines traditional tender management functionality with advanced AI-powered anomaly detection to identify potentially suspicious bidding patterns and maintain accountability through comprehensive audit logging.

The platform serves as a centralized hub where government departments can publish tenders, companies can submit bids, and administrators can monitor the entire process for irregularities using machine learning algorithms.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
The system uses a **Flask-based REST API** architecture with modular service components. The main application (`app.py`) serves as the central controller that orchestrates various specialized services:

- **Database Service**: SQLite-based data persistence layer handling tenders, bids, and audit logs
- **ML Service**: Isolation Forest-based anomaly detection system for identifying suspicious bid patterns
- **Chatbot Service**: OpenAI GPT integration with FAQ fallback for user assistance
- **NLP Service**: Text analysis service using spaCy for proposal evaluation
- **File Handler**: Secure file upload and validation system

### Frontend Architecture
The system uses a **static HTML/CSS/JavaScript frontend** with Tailwind CSS for styling and Chart.js for data visualization. The frontend communicates with the backend through RESTful API calls using fetch requests.

### Data Storage Solutions
**SQLite Database** serves as the primary data store with three main tables:
- **Tenders**: Store tender information including metadata and file references
- **Bids**: Store bid submissions with anomaly scores and suspicious flags
- **Audit Logs**: Track all system actions for transparency and accountability

File storage uses a **local filesystem approach** with uploads stored in `/uploads` directory and trained ML models in `/models` directory.

### Machine Learning Architecture
The system employs **Isolation Forest algorithm** from scikit-learn for unsupervised anomaly detection with:
- 10% contamination threshold for flagging suspicious bids
- Feature extraction from bid amounts, proposal text, timing patterns, and company information
- Model persistence using joblib for training/retraining capabilities
- Performance metrics tracking for model evaluation

### Security and Validation
Multi-layered security approach includes:
- **File validation**: Size limits (15MB), extension filtering, and content verification
- **Hash verification**: SHA-256 hashing for file integrity
- **Audit logging**: Comprehensive action tracking for accountability
- **Input sanitization**: Secure filename handling and request validation

### API Design
RESTful API structure with JSON responses:
- `/api/dashboard` - System statistics and metrics
- `/api/tenders` - Tender CRUD operations
- `/api/bids` - Bid submission and retrieval
- `/api/bids/suspicious` - Anomaly detection triggers
- `/api/alerts` - Alert management
- `/api/chat` - Chatbot interactions
- `/api/model/*` - ML model management

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web framework for REST API development
- **Flask-CORS**: Cross-origin resource sharing for frontend integration
- **SQLite3**: Embedded database for data persistence

### Machine Learning and AI
- **scikit-learn**: Machine learning library for Isolation Forest algorithm
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing support
- **spaCy**: Natural language processing for text analysis
- **joblib**: Model serialization and persistence

### External APIs
- **OpenAI GPT API**: Chatbot service integration for intelligent user assistance
- Requires `OPENAI_API_KEY` environment variable for functionality

### File Processing
- **Werkzeug**: Secure file handling and upload utilities
- **hashlib**: File integrity verification through SHA-256 hashing

### Development and Utilities
- **python-dotenv**: Environment variable management
- **datetime**: Timestamp handling for audit trails

The system is designed to be self-contained with minimal external service dependencies, relying primarily on local processing and storage with optional cloud AI integration for enhanced chatbot capabilities.