import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

class Database:
    def __init__(self, db_path: str = 'actms.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create tenders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                department TEXT NOT NULL,
                estimated_value REAL NOT NULL,
                deadline TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                file_hash TEXT
            )
        ''')
        
        # Create bids table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_id INTEGER NOT NULL,
                company_name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                bid_amount REAL NOT NULL,
                proposal TEXT NOT NULL,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                anomaly_score REAL DEFAULT 0.0,
                is_suspicious BOOLEAN DEFAULT 0,
                file_path TEXT,
                file_hash TEXT,
                FOREIGN KEY (tender_id) REFERENCES tenders (id)
            )
        ''')
        
        # Create audit_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        ''')
        
        # Create alerts table for AI alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bid_id INTEGER NOT NULL,
                alert_type TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY (bid_id) REFERENCES bids (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_action(self, entity_type: str, entity_id: int, action: str, details: Optional[str] = None):
        """Log an action to the audit log"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_logs (entity_type, entity_id, action, details)
            VALUES (?, ?, ?, ?)
        ''', (entity_type, entity_id, action, details))
        
        conn.commit()
        conn.close()
    
    # Tender CRUD operations
    def create_tender(self, title: str, description: str, department: str, 
                     estimated_value: float, deadline: str, file_path: Optional[str] = None, 
                     file_hash: Optional[str] = None) -> int:
        """Create a new tender"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tenders (title, description, department, estimated_value, deadline, file_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (title, description, department, estimated_value, deadline, file_path, file_hash))
        
        tender_id = cursor.lastrowid
        if tender_id is None:
            raise Exception("Failed to create tender")
        conn.commit()
        conn.close()
        
        # Log the action
        self.log_action('tender', tender_id, 'created', f'Tender "{title}" created')
        
        return tender_id
    
    def get_all_tenders(self) -> List[Dict]:
        """Get all tenders"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tenders ORDER BY created_at DESC')
        tenders = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return tenders
    
    def get_tender_by_id(self, tender_id: int) -> Optional[Dict]:
        """Get a tender by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tenders WHERE id = ?', (tender_id,))
        tender = cursor.fetchone()
        
        conn.close()
        return dict(tender) if tender else None
    
    def get_tenders_by_status(self, status: str):
        """Get tenders by status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tenders WHERE status = ? ORDER BY created_at DESC', (status,))
        tenders = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return tenders
    
    def update_tender_status(self, tender_id: int, status: str):
        """Update tender status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE tenders SET status = ? WHERE id = ?', (status, tender_id))
        conn.commit()
        conn.close()
        
        self.log_action('tender', tender_id, 'status_updated', f'Status changed to {status}')
    
    # Bid CRUD operations
    def create_bid(self, tender_id: int, company_name: str, contact_email: str,
                   bid_amount: float, proposal: str, file_path: Optional[str] = None,
                   file_hash: Optional[str] = None) -> int:
        """Create a new bid"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bids (tender_id, company_name, contact_email, bid_amount, proposal, file_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tender_id, company_name, contact_email, bid_amount, proposal, file_path, file_hash))
        
        bid_id = cursor.lastrowid
        if bid_id is None:
            raise Exception("Failed to create bid")
        conn.commit()
        conn.close()
        
        # Log the action
        self.log_action('bid', bid_id, 'submitted', f'Bid submitted by {company_name} for tender {tender_id}')
        
        return bid_id
    
    def get_all_bids(self) -> List[Dict]:
        """Get all bids with tender information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT b.*, t.title as tender_title, t.department as tender_department
            FROM bids b
            JOIN tenders t ON b.tender_id = t.id
            ORDER BY b.submitted_at DESC
        ''')
        bids = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return bids
    
    def get_bids_by_tender(self, tender_id: int) -> List[Dict]:
        """Get all bids for a specific tender"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bids WHERE tender_id = ? ORDER BY submitted_at DESC', (tender_id,))
        bids = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return bids
    
    def update_bid_anomaly_score(self, bid_id: int, anomaly_score: float, is_suspicious: bool):
        """Update bid anomaly score and suspicious flag"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bids SET anomaly_score = ?, is_suspicious = ? WHERE id = ?
        ''', (anomaly_score, is_suspicious, bid_id))
        
        conn.commit()
        conn.close()
        
        if is_suspicious:
            self.log_action('bid', bid_id, 'flagged_suspicious', f'Anomaly score: {anomaly_score:.3f}')
    
    def get_suspicious_bids(self) -> List[Dict]:
        """Get all suspicious bids"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT b.*, t.title as tender_title, t.department as tender_department
            FROM bids b
            JOIN tenders t ON b.tender_id = t.id
            WHERE b.is_suspicious = 1
            ORDER BY b.anomaly_score DESC
        ''')
        bids = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return bids
    
    # Alert operations
    def create_alert(self, bid_id: int, alert_type: str, description: str, severity: str = 'medium') -> int:
        """Create a new alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (bid_id, alert_type, description, severity)
            VALUES (?, ?, ?, ?)
        ''', (bid_id, alert_type, description, severity))
        
        alert_id = cursor.lastrowid
        if alert_id is None:
            raise Exception("Failed to create alert")
        conn.commit()
        conn.close()
        
        self.log_action('alert', alert_id, 'created', f'Alert created for bid {bid_id}: {description}')
        
        return alert_id
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.*, b.company_name, b.bid_amount, t.title as tender_title
            FROM alerts a
            JOIN bids b ON a.bid_id = b.id
            JOIN tenders t ON b.tender_id = t.id
            WHERE a.status = 'active'
            ORDER BY a.created_at DESC
        ''')
        alerts = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return alerts
    
    def resolve_alert(self, alert_id: int):
        """Resolve an alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()
        
        self.log_action('alert', alert_id, 'resolved', 'Alert marked as resolved')
    
    # Dashboard statistics
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get total tenders
        cursor.execute('SELECT COUNT(*) FROM tenders')
        total_tenders = cursor.fetchone()[0]
        
        # Get active tenders
        cursor.execute('SELECT COUNT(*) FROM tenders WHERE status = ?', ('active',))
        active_tenders = cursor.fetchone()[0]
        
        # Get total bids
        cursor.execute('SELECT COUNT(*) FROM bids')
        total_bids = cursor.fetchone()[0]
        
        # Get suspicious bids
        cursor.execute('SELECT COUNT(*) FROM bids WHERE is_suspicious = 1')
        suspicious_bids = cursor.fetchone()[0]
        
        # Get active alerts
        cursor.execute('SELECT COUNT(*) FROM alerts WHERE status = ?', ('active',))
        active_alerts = cursor.fetchone()[0]
        
        # Get recent activity (last 7 days)
        cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE timestamp >= datetime('now', '-7 days')")
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_tenders': total_tenders,
            'active_tenders': active_tenders,
            'total_bids': total_bids,
            'suspicious_bids': suspicious_bids,
            'active_alerts': active_alerts,
            'recent_activity': recent_activity
        }
    
    def get_audit_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent audit logs"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?', (limit,))
        logs = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return logs