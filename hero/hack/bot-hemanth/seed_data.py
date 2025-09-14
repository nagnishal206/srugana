#!/usr/bin/env python3
"""
Sample data seeder for ACTMS demonstration purposes
This script populates the database with sample tenders and bids to demonstrate the system functionality.
"""

import random
from datetime import datetime, timedelta
from database import Database

def seed_sample_data():
    """Populate the database with sample data for demonstration"""
    db = Database()
    
    print("🌱 Starting database seeding...")
    
    # Sample departments
    departments = [
        "Transportation", "Infrastructure", "Healthcare", "Education", 
        "Technology", "Public Works", "Defense", "Energy"
    ]
    
    # Sample company names
    companies = [
        "TechCorp Solutions", "BuildRight Construction", "MediSupply Co",
        "EduTech Systems", "GreenEnergy Ltd", "SecureNet Inc",
        "UrbanDev Group", "LogiTrans Services", "DataFlow Systems",
        "CleanWater Corp", "SmartCity Solutions", "InnovateBuild",
        "HealthTech Plus", "EcoConstruct", "DigitalBridge Co"
    ]
    
    # Create sample tenders
    print("Creating sample tenders...")
    tender_ids = []
    
    for i in range(8):
        title = f"Public {random.choice(['Infrastructure', 'Technology', 'Healthcare', 'Transportation'])} Project {i+1}"
        description = f"Comprehensive {random.choice(['development', 'upgrade', 'implementation'])} project for public sector"
        department = random.choice(departments)
        estimated_value = random.randint(50000, 500000)
        
        # Create deadline 1-6 months in the future
        deadline_date = datetime.now() + timedelta(days=random.randint(30, 180))
        deadline = deadline_date.strftime('%Y-%m-%d')
        
        tender_id = db.create_tender(
            title=title,
            description=description,
            department=department,
            estimated_value=estimated_value,
            deadline=deadline
        )
        tender_ids.append(tender_id)
        print(f"✓ Created tender: {title}")
    
    # Create sample bids
    print("Creating sample bids...")
    
    for i in range(25):  # Create 25 bids to ensure we have enough for ML
        tender_id = random.choice(tender_ids)
        company_name = random.choice(companies)
        contact_email = f"contact@{company_name.lower().replace(' ', '').replace('.', '')}.com"
        
        # Get tender estimated value to create realistic bids
        tender = db.get_tender_by_id(tender_id)
        if tender:
            estimated_value = tender['estimated_value']
            # Bids typically range from 80% to 120% of estimated value
            bid_variation = random.uniform(0.8, 1.2)
            # Add some outliers for anomaly detection
            if random.random() < 0.15:  # 15% chance of outlier
                bid_variation = random.uniform(0.4, 1.8)  # More extreme variation
            
            bid_amount = estimated_value * bid_variation
        else:
            bid_amount = random.randint(40000, 600000)
        
        # Generate proposal text
        proposals = [
            "We propose a comprehensive solution with advanced methodologies and proven track record.",
            "Our team brings extensive experience and innovative approaches to deliver exceptional results.",
            "We offer competitive pricing with superior quality and timely delivery guarantees.",
            "Our proposal includes cutting-edge technology and best practices in the industry.",
            "We provide end-to-end services with dedicated project management and support.",
            "Our solution focuses on sustainability, efficiency, and long-term value creation.",
            "We bring specialized expertise and strategic partnerships to ensure project success.",
            "Our approach emphasizes quality assurance, risk mitigation, and stakeholder satisfaction."
        ]
        
        proposal = random.choice(proposals)
        if random.random() < 0.1:  # 10% chance of very short proposal (potential flag)
            proposal = "Quick solution. Low cost."
        
        # Create submission time (last 30 days)
        submitted_at = datetime.now() - timedelta(days=random.randint(1, 30))
        
        bid_id = db.create_bid(
            tender_id=tender_id,
            company_name=company_name,
            contact_email=contact_email,
            bid_amount=bid_amount,
            proposal=proposal
        )
        
        print(f"✓ Created bid {i+1}: {company_name} - ${bid_amount:,.0f}")
    
    # Get final counts
    tenders = db.get_all_tenders()
    bids = db.get_all_bids()
    
    print(f"\n🎉 Seeding completed!")
    print(f"📊 Summary:")
    print(f"   • Tenders: {len(tenders)}")
    print(f"   • Bids: {len(bids)}")
    print(f"   • Ready for ML anomaly detection: {'Yes' if len(bids) >= 10 else 'No'}")
    print(f"\n💡 You can now view the dashboard with fully functional charts and analytics!")

if __name__ == "__main__":
    seed_sample_data()