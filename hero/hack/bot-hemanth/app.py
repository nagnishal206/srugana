from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
from datetime import datetime

# Import our custom services
from database import Database
from utils.file_handler import FileHandler
from services.ml_service import MLAnomalyDetector
from services.chatbot_service import ChatbotService
from services.nlp_service import NLPService

app = Flask(__name__)

# Production-ready configuration
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

# Configure CORS based on environment
if FLASK_ENV == 'production':
    # Secure CORS configuration for production
    CORS(app, origins=CORS_ORIGINS, supports_credentials=True)
else:
    # More permissive CORS for development
    CORS(app)  # Enable CORS for frontend integration

# Initialize services
db = Database()
file_handler = FileHandler()
ml_detector = MLAnomalyDetector()
chatbot = ChatbotService()
nlp_service = NLPService()

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'attached_assets/index_1757831615725.html')

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        stats = db.get_dashboard_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tenders', methods=['GET', 'POST'])
def handle_tenders():
    """Handle tender operations"""
    if request.method == 'GET':
        try:
            # Check for status query parameter
            status = request.args.get('status')
            if status:
                tenders = db.get_tenders_by_status(status)
            else:
                tenders = db.get_all_tenders()
            
            return jsonify({
                'success': True,
                'data': tenders
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    elif request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title')
            description = request.form.get('description')
            department = request.form.get('department')
            estimated_value = float(request.form.get('estimated_value', 0))
            deadline = request.form.get('deadline')
            
            # Validate required fields
            if not all([title, description, department, deadline]):
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields'
                }), 400
            
            # Handle file upload
            file_path = None
            file_hash = None
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename:
                    file_path, file_hash, error = file_handler.save_file(file, 'tender')
                    if error:
                        return jsonify({
                            'success': False,
                            'error': f'File upload error: {error}'
                        }), 400
            
            # Create tender
            tender_id = db.create_tender(
                title=title,
                description=description,
                department=department,
                estimated_value=estimated_value,
                deadline=deadline,
                file_path=file_path,
                file_hash=file_hash
            )
            
            return jsonify({
                'success': True,
                'data': {
                    'tender_id': tender_id,
                    'message': 'Tender created successfully'
                }
            })
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': 'Invalid estimated value'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/api/bids', methods=['GET', 'POST'])
def handle_bids():
    """Handle bid operations"""
    if request.method == 'GET':
        try:
            bids = db.get_all_bids()
            return jsonify({
                'success': True,
                'data': bids
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    elif request.method == 'POST':
        try:
            # Get form data
            tender_id = int(request.form.get('tender_id'))
            company_name = request.form.get('company_name')
            contact_email = request.form.get('contact_email')
            bid_amount = float(request.form.get('bid_amount', 0))
            proposal = request.form.get('proposal', '')
            
            # Validate required fields
            if not all([tender_id, company_name, contact_email, proposal]):
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields'
                }), 400
            
            # Validate tender exists
            tender = db.get_tender_by_id(tender_id)
            if not tender:
                return jsonify({
                    'success': False,
                    'error': 'Invalid tender ID'
                }), 400
            
            # Handle file upload
            file_path = None
            file_hash = None
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename:
                    file_path, file_hash, error = file_handler.save_file(file, 'bid')
                    if error:
                        return jsonify({
                            'success': False,
                            'error': f'File upload error: {error}'
                        }), 400
            
            # Create bid
            bid_id = db.create_bid(
                tender_id=tender_id,
                company_name=company_name,
                contact_email=contact_email,
                bid_amount=bid_amount,
                proposal=proposal,
                file_path=file_path,
                file_hash=file_hash
            )
            
            return jsonify({
                'success': True,
                'data': {
                    'bid_id': bid_id,
                    'message': 'Bid submitted successfully'
                }
            })
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': 'Invalid numeric values'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/api/bids/suspicious', methods=['GET'])
def get_suspicious_bids():
    """Run anomaly detection and get suspicious bids"""
    try:
        # Get all bids for analysis
        all_bids = db.get_all_bids()
        
        if len(all_bids) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for anomaly detection. Need at least 10 bids.',
                'data': []
            })
        
        try:
            # Run anomaly detection
            anomaly_results = ml_detector.predict_anomalies(all_bids)
            
            # Update database with anomaly scores
            for result in anomaly_results:
                db.update_bid_anomaly_score(
                    bid_id=result['bid_id'],
                    anomaly_score=result['anomaly_score'],
                    is_suspicious=result['is_suspicious']
                )
                
                # Create alerts for suspicious bids
                if result['is_suspicious']:
                    alert_description = f"Suspicious bid detected. Reasons: {', '.join(result['reasons'])}"
                    db.create_alert(
                        bid_id=result['bid_id'],
                        alert_type='anomaly_detection',
                        description=alert_description,
                        severity='high' if result['anomaly_score'] > 0.8 else 'medium'
                    )
            
            # Get updated suspicious bids
            suspicious_bids = db.get_suspicious_bids()
            
            return jsonify({
                'success': True,
                'data': suspicious_bids,
                'analysis_results': anomaly_results
            })
            
        except Exception as ml_error:
            # If ML fails, still return suspicious bids from database
            suspicious_bids = db.get_suspicious_bids()
            return jsonify({
                'success': True,
                'data': suspicious_bids,
                'warning': f'ML analysis failed: {str(ml_error)}. Showing previously flagged bids.'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all active alerts"""
    try:
        alerts = db.get_active_alerts()
        return jsonify({
            'success': True,
            'data': alerts
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts/<int:alert_id>/resolve', methods=['PUT'])
def resolve_alert(alert_id):
    """Mark an alert as resolved"""
    try:
        db.resolve_alert(alert_id)
        return jsonify({
            'success': True,
            'message': 'Alert resolved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chatbot', methods=['POST'])
def chat():
    """Handle chatbot interactions"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        # Get chatbot response
        response_data = chatbot.get_response(user_message)
        
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'source': response_data['source'],
            'confidence': response_data['confidence']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'I apologize, but I encountered an error. Please try again.',
            'source': 'error',
            'confidence': 0.0
        }), 500

@app.route('/api/ml/train', methods=['POST'])
def train_model():
    """Train/retrain the ML model"""
    try:
        # Get all bids for training
        all_bids = db.get_all_bids()
        
        if len(all_bids) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training. Need at least 10 bids.'
            })
        
        # Train the model
        result = ml_detector.train_model(all_bids)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/test', methods=['GET'])
def test_model():
    """Test model performance"""
    try:
        # Get test data (all bids)
        test_bids = db.get_all_bids()
        
        # Get known suspicious bid IDs
        suspicious_bids = db.get_suspicious_bids()
        known_suspicious_ids = [bid['id'] for bid in suspicious_bids]
        
        # Test the model
        result = ml_detector.test_model(test_bids, known_suspicious_ids)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        info = ml_detector.get_model_info()
        return jsonify({
            'success': True,
            'data': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/audit-logs', methods=['GET'])
def get_audit_logs():
    """Get audit logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = db.get_audit_logs(limit)
        return jsonify({
            'success': True,
            'data': logs
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text using NLP service"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        analysis = nlp_service.analyze_text(text)
        
        return jsonify({
            'success': True,
            'data': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Initialize the database
    print("Initializing ACTMS system...")
    print(f"Environment: {FLASK_ENV}")
    print(f"Debug mode: {DEBUG_MODE}")
    print(f"Database initialized: {os.path.exists('actms.db')}")
    print(f"Upload directory: {os.path.exists('uploads')}")
    print(f"Models directory: {os.path.exists('models')}")
    print(f"Chatbot AI available: {chatbot.is_ai_available()}")
    
    # Get port and host from environment variables
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Production readiness warnings
    if FLASK_ENV == 'production' and DEBUG_MODE:
        print("WARNING: Debug mode is enabled in production environment!")
    
    if FLASK_ENV == 'production':
        print("PRODUCTION MODE: Using secure CORS configuration")
        print(f"CORS Origins: {CORS_ORIGINS}")
    
    # Start the Flask server
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)