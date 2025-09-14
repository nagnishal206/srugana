import json
import os
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types

class ChatbotService:
    def __init__(self, faq_file: str = 'faq.json'):
        self.faq_file = faq_file
        self.faqs = self.load_faqs()
        self.gemini_client = None
        self.gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')  # Default to universally available model
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Gemini client"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            print("Warning: GEMINI_API_KEY not found. Chatbot will only use FAQs.")
    
    def load_faqs(self) -> List[Dict[str, str]]:
        """Load FAQs from JSON file"""
        if os.path.exists(self.faq_file):
            try:
                with open(self.faq_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Default FAQs if file doesn't exist
        default_faqs = [
            {
                "question": "What is ACTMS?",
                "answer": "ACTMS (Anti-Corruption Tender Management System) is a comprehensive platform designed to ensure transparency and integrity in tender processes. It uses AI-powered analysis to detect suspicious bidding patterns and prevent corruption."
            },
            {
                "question": "How does the anomaly detection work?",
                "answer": "Our system uses advanced machine learning algorithms, specifically Isolation Forest, to analyze bid submissions. It examines factors like bid amounts, proposal quality, submission timing, and company patterns to identify potentially suspicious activities."
            },
            {
                "question": "What file formats are supported for uploads?",
                "answer": "The system supports PDF, DOC, DOCX, TXT, and RTF file formats. All files must be under 15MB in size and undergo security validation before being stored."
            },
            {
                "question": "How can I submit a bid?",
                "answer": "To submit a bid, navigate to the 'Bid Submission' section, select the tender you're interested in, fill out the required information including company details and bid amount, upload your proposal document, and submit. Your bid will be automatically analyzed for compliance."
            },
            {
                "question": "What happens if my bid is flagged as suspicious?",
                "answer": "If a bid is flagged, it undergoes additional review by administrators. This doesn't mean the bid is invalid - the system may flag bids due to statistical anomalies. You can contact support to provide additional clarification if needed."
            },
            {
                "question": "How is my data protected?",
                "answer": "We implement multiple security measures including file validation, hash verification, audit logging, and secure data storage. All actions are logged for transparency and accountability."
            },
            {
                "question": "Can I track the status of my bid?",
                "answer": "Yes, you can view the status of your submitted bids in the dashboard. You'll see information about submission status, any alerts, and the review process."
            },
            {
                "question": "What is the audit log?",
                "answer": "The audit log tracks all major actions in the system including tender creation, bid submissions, status changes, and alert resolutions. This ensures full transparency and accountability in the tender process."
            },
            {
                "question": "How do I report a problem or get support?",
                "answer": "You can use this chatbot for immediate assistance, or contact the system administrators through the contact information provided in your tender documentation."
            },
            {
                "question": "What tender information is publicly available?",
                "answer": "Tender titles, descriptions, departments, estimated values, and deadlines are publicly visible. Bid details and company information are kept confidential and only accessible to authorized personnel."
            }
        ]
        
        # Save default FAQs
        self.save_faqs(default_faqs)
        return default_faqs
    
    def save_faqs(self, faqs: List[Dict[str, str]]):
        """Save FAQs to JSON file"""
        try:
            with open(self.faq_file, 'w', encoding='utf-8') as f:
                json.dump(faqs, f, indent=2, ensure_ascii=False)
        except IOError:
            pass
    
    def find_faq_answer(self, user_message: str) -> Optional[str]:
        """Find matching FAQ answer based on user message"""
        user_message_lower = user_message.lower()
        
        # Simple keyword matching for FAQs
        for faq in self.faqs:
            question_lower = faq['question'].lower()
            answer_lower = faq['answer'].lower()
            
            # Check if key words from question appear in user message
            question_words = set(question_lower.split())
            user_words = set(user_message_lower.split())
            
            # Calculate word overlap
            overlap = question_words.intersection(user_words)
            overlap_ratio = len(overlap) / len(question_words) if question_words else 0
            
            # Also check if user message appears in question or answer
            if (overlap_ratio > 0.3 or 
                any(word in user_message_lower for word in question_lower.split() if len(word) > 3) or
                user_message_lower in question_lower or
                question_lower in user_message_lower):
                return faq['answer']
        
        return None
    
    def get_gemini_response(self, user_message: str, context: Optional[str] = None) -> str:
        """Get response from Gemini model"""
        if not self.gemini_client:
            return "I'm sorry, but I don't have access to advanced AI features right now. Please check our FAQ section or contact support for assistance."
        
        try:
            # Build system message with context about ACTMS
            system_instruction = """You are a helpful assistant for ACTMS (Anti-Corruption Tender Management System). 
            You help users understand the tender management process, bid submission, and system features.
            
            Key information about ACTMS:
            - It's an anti-corruption tender management system
            - Uses AI/ML for detecting suspicious bidding patterns
            - Supports file uploads (PDF, DOC, DOCX, TXT, RTF)
            - Has features for tender management, bid submission, AI analysis, and dashboard analytics
            - Maintains audit logs for transparency
            - Uses Isolation Forest algorithm for anomaly detection
            
            Be helpful, professional, and provide accurate information about the system.
            If asked about technical details you're not certain about, suggest contacting system administrators.
            """
            
            if context:
                system_instruction += f"\n\nAdditional context: {context}"
            
            # Build the prompt for Gemini
            prompt = f"{system_instruction}\n\nUser question: {user_message}"
            
            # Use configurable Gemini model from environment variable
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            
            return response.text.strip() if response.text else "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            return "I apologize, but I'm experiencing some technical difficulties. Please try again later or contact support."    
    
    def get_response(self, user_message: str, use_ai: bool = True, context: Optional[str] = None) -> Dict[str, Any]:
        """Get chatbot response, trying FAQ first, then Gemini if enabled"""
        if not user_message or not user_message.strip():
            return {
                'response': "Please ask me a question about the tender management system!",
                'source': 'system',
                'confidence': 1.0
            }
        
        # First, try to find answer in FAQs
        faq_answer = self.find_faq_answer(user_message)
        if faq_answer:
            return {
                'response': faq_answer,
                'source': 'faq',
                'confidence': 0.8
            }
        
        # If no FAQ match and AI is enabled, use Gemini
        if use_ai and self.gemini_client:
            ai_response = self.get_gemini_response(user_message, context)
            return {
                'response': ai_response,
                'source': 'ai',
                'confidence': 0.9
            }
        
        # Fallback response
        return {
            'response': "I don't have a specific answer for that question. Please check our documentation or contact system support for more detailed assistance.",
            'source': 'fallback',
            'confidence': 0.3
        }
    
    def add_faq(self, question: str, answer: str) -> bool:
        """Add a new FAQ entry"""
        try:
            new_faq = {
                'question': question.strip(),
                'answer': answer.strip()
            }
            self.faqs.append(new_faq)
            self.save_faqs(self.faqs)
            return True
        except Exception:
            return False
    
    def update_faq(self, index: int, question: str, answer: str) -> bool:
        """Update an existing FAQ entry"""
        try:
            if 0 <= index < len(self.faqs):
                self.faqs[index] = {
                    'question': question.strip(),
                    'answer': answer.strip()
                }
                self.save_faqs(self.faqs)
                return True
            return False
        except Exception:
            return False
    
    def delete_faq(self, index: int) -> bool:
        """Delete an FAQ entry"""
        try:
            if 0 <= index < len(self.faqs):
                del self.faqs[index]
                self.save_faqs(self.faqs)
                return True
            return False
        except Exception:
            return False
    
    def get_all_faqs(self) -> List[Dict[str, str]]:
        """Get all FAQ entries"""
        return self.faqs.copy()
    
    def search_faqs(self, query: str) -> List[Dict[str, Any]]:
        """Search FAQs by keyword"""
        query_lower = query.lower()
        results = []
        
        for i, faq in enumerate(self.faqs):
            score = 0
            question_lower = faq['question'].lower()
            answer_lower = faq['answer'].lower()
            
            # Check for exact matches
            if query_lower in question_lower:
                score += 10
            if query_lower in answer_lower:
                score += 5
            
            # Check for word matches
            query_words = set(query_lower.split())
            question_words = set(question_lower.split())
            answer_words = set(answer_lower.split())
            
            question_overlap = query_words.intersection(question_words)
            answer_overlap = query_words.intersection(answer_words)
            
            score += len(question_overlap) * 2
            score += len(answer_overlap)
            
            if score > 0:
                results.append({
                    'index': i,
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'relevance_score': score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:5]  # Return top 5 results
    
    def get_conversation_context(self, recent_messages: List[str]) -> str:
        """Build context from recent conversation messages"""
        if not recent_messages:
            return ""
        
        # Take last 3 messages for context
        context_messages = recent_messages[-3:]
        return "Recent conversation: " + " | ".join(context_messages)
    
    def is_ai_available(self) -> bool:
        """Check if AI features are available"""
        return self.gemini_client is not None