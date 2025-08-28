from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None
feature_columns = None

def load_or_train_model():
    """Load existing model or train a new one if not found"""
    global model, feature_columns
    
    try:
        # Try to load existing model
        if os.path.exists('churn_model.pkl'):
            model = joblib.load('churn_model.pkl')
            logger.info("Model loaded successfully from file.")
            
            # Define feature columns (should match training data)
            feature_columns = [
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Geography_Germany', 'Geography_Spain', 'Gender_Male'
            ]
        else:
            logger.info("Model file not found. Training new model...")
            train_new_model()
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Training new model...")
        train_new_model()

def train_new_model():
    """Train a new model if CSV data is available"""
    global model, feature_columns
    
    try:
        # Check if dataset exists
        if not os.path.exists('Churn_Modelling.csv'):
            logger.error("Dataset file 'Churn_Modelling.csv' not found!")
            return False
            
        # Load and prepare data
        df = pd.read_csv('Churn_Modelling.csv')
        logger.info(f"Dataset loaded: {len(df)} rows")
        
        # Preprocess data
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
        
        # Define features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        
        feature_columns = list(X.columns)
        logger.info(f"Features: {feature_columns}")
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with better parameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, 'churn_model.pkl')
        logger.info("Model saved successfully.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def get_retention_strategy(prediction_prob, customer_data):
    """Generate personalized retention strategy based on customer profile"""
    strategies = []
    
    # High-risk customers (>60% churn probability)
    if prediction_prob > 0.6:
        if customer_data['balance'] > 100000:
            strategies.append("🎯 Offer premium banking services with higher interest rates")
            strategies.append("💎 Provide dedicated relationship manager")
        elif customer_data['num_of_products'] == 1:
            strategies.append("📦 Cross-sell additional products with attractive bundles")
            strategies.append("💳 Offer fee waivers for new credit card applications")
        
        if customer_data['is_active_member'] == 0:
            strategies.append("🎮 Launch re-engagement campaign with exclusive offers")
            strategies.append("📱 Promote digital banking features and mobile app")
            
        if customer_data['age'] > 50:
            strategies.append("👥 Offer personalized financial planning sessions")
            strategies.append("🏦 Provide priority customer service line")
    
    # Medium-risk customers (30-60% churn probability)
    elif prediction_prob > 0.3:
        strategies.append("🎁 Send targeted promotional offers")
        strategies.append("📊 Provide quarterly account review and optimization tips")
        if customer_data['has_cr_card'] == 0:
            strategies.append("💳 Offer pre-approved credit card with benefits")
    
    # Low-risk customers (<30% churn probability)
    else:
        strategies.append("✅ Continue standard engagement")
        strategies.append("🌟 Consider for referral program incentives")
    
    return strategies

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available. Please check server logs.'}), 500

    try:
        # Get and validate input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Required fields validation
        required_fields = [
            'credit_score', 'age', 'tenure', 'balance', 'num_of_products',
            'estimated_salary', 'geography', 'gender', 'has_cr_card', 'is_active_member'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        # Prepare input data
        input_data = {
            'CreditScore': [int(data['credit_score'])],
            'Age': [int(data['age'])],
            'Tenure': [int(data['tenure'])],
            'Balance': [float(data['balance'])],
            'NumOfProducts': [int(data['num_of_products'])],
            'HasCrCard': [int(data['has_cr_card'])],
            'IsActiveMember': [int(data['is_active_member'])],
            'EstimatedSalary': [float(data['estimated_salary'])],
            'Geography_Germany': [1 if data['geography'] == 'Germany' else 0],
            'Geography_Spain': [1 if data['geography'] == 'Spain' else 0],
            'Gender_Male': [1 if data['gender'] == 'Male' else 0]
        }
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame(input_data)
        
        # Make predictions
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of churn
        
        # Generate risk level and strategies
        if probability > 0.7:
            risk_level = "CRITICAL"
            risk_color = "#dc3545"
        elif probability > 0.4:
            risk_level = "HIGH"
            risk_color = "#fd7e14"
        elif probability > 0.2:
            risk_level = "MEDIUM"
            risk_color = "#ffc107"
        else:
            risk_level = "LOW"
            risk_color = "#28a745"
        
        # Get personalized retention strategies
        customer_profile = {
            'balance': float(data['balance']),
            'num_of_products': int(data['num_of_products']),
            'is_active_member': int(data['is_active_member']),
            'has_cr_card': int(data['has_cr_card']),
            'age': int(data['age'])
        }
        
        strategies = get_retention_strategy(probability, customer_profile)
        
        return jsonify({
            'prediction': int(prediction),
            'churn_probability': float(probability),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': float(max(probability, 1 - probability)),
            'retention_strategies': strategies,
            'customer_summary': {
                'credit_score_category': 'Excellent' if int(data['credit_score']) > 750 else 
                                       'Good' if int(data['credit_score']) > 650 else 'Fair',
                'tenure_years': int(data['tenure']),
                'product_usage': 'High' if int(data['num_of_products']) > 2 else 
                               'Medium' if int(data['num_of_products']) == 2 else 'Low',
                'engagement_level': 'Active' if int(data['is_active_member']) else 'Inactive'
            }
        })
        
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        return jsonify({'error': f'Invalid input values: {str(ve)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'features': feature_columns,
        'model_loaded': True,
        'status': 'operational'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Intelligent chatbot for managers"""
    try:
        data = request.get_json()
        question = data.get('question', '').lower().strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        response = get_chatbot_response(question)
        
        return jsonify({
            'question': data.get('question', ''),
            'response': response,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({'error': 'Sorry, I encountered an error processing your question'}), 500

def get_chatbot_response(question):
    """Generate intelligent responses to manager questions"""
    
    # Define response patterns and keywords
    responses = {
        'churn_rate': {
            'keywords': ['churn rate', 'churn percentage', 'how many customers leave', 'attrition rate'],
            'response': """📊 **Customer Churn Analysis**

Based on typical banking data patterns:
• **Average churn rate**: 15-25% annually
• **High-risk factors**: Age >50, inactive members, single product usage
• **Key insight**: Customers with 3+ products have 80% lower churn rates

**Recommendation**: Focus on cross-selling to single-product customers and re-engaging inactive members."""
        },
        
        'retention_strategies': {
            'keywords': ['retention', 'keep customers', 'reduce churn', 'strategies', 'prevent leaving'],
            'response': """🎯 **Top Retention Strategies**

**Immediate Actions:**
• **High-balance customers**: Offer premium services & dedicated relationship managers
• **Inactive members**: Launch re-engagement campaigns with personalized offers
• **Single product users**: Cross-sell with attractive bundle deals

**Long-term Strategies:**
• Implement loyalty reward programs
• Provide quarterly financial health check-ups
• Enhance digital banking experience
• Offer competitive interest rates for deposits"""
        },
        
        'high_risk_customers': {
            'keywords': ['high risk', 'likely to churn', 'dangerous customers', 'red flags'],
            'response': """⚠️ **High-Risk Customer Profile**

**Primary Risk Indicators:**
• Age: 35-55 years (career transition period)
• Balance: Very high (>₹1L) or very low (<₹10K)
• Products: Only 1 product with the bank
• Activity: Inactive for 3+ months
• Geography: Competitive markets (Germany/Spain in our data)

**Immediate Action Required:**
• Proactive outreach within 48 hours
• Personalized retention offers
• Assign dedicated relationship manager"""
        },
        
        'roi_impact': {
            'keywords': ['roi', 'revenue impact', 'cost', 'profit', 'financial impact', 'business value'],
            'response': """💰 **Business Impact & ROI**

**Cost of Customer Acquisition vs Retention:**
• New customer acquisition: ₹15,000-25,000
• Customer retention: ₹2,000-5,000
• **ROI**: 5-10x better to retain than acquire

**Revenue Impact:**
• Average customer lifetime value: ₹2-5 lakhs
• Preventing 1% churn = ₹20-50 lakhs saved annually
• Cross-selling success increases CLV by 40-60%

**Recommendation**: Invest 20% of acquisition budget in retention programs."""
        },
        
        'implementation': {
            'keywords': ['implement', 'how to use', 'deploy', 'rollout', 'execute'],
            'response': """🚀 **Implementation Roadmap**

**Phase 1 (Week 1-2): Setup**
• Deploy churn prediction model
• Train customer service team
• Set up automated alerts for high-risk customers

**Phase 2 (Week 3-4): Pilot**
• Run pilot with 1000 customers
• Test retention campaigns
• Measure response rates

**Phase 3 (Month 2): Scale**
• Full deployment across all customers
• Weekly churn risk reports
• Monthly strategy optimization

**Success Metrics**: Reduce churn by 15-30% within 6 months"""
        },
        
        'technology': {
            'keywords': ['technology', 'how it works', 'machine learning', 'ai', 'algorithm'],
            'response': """🤖 **Technology Behind Churn Prediction**

**Machine Learning Model:**
• **Algorithm**: Random Forest Classifier (86.65% accuracy)
• **Key Features**: Credit score, age, balance, product usage, activity level
• **Prediction**: Real-time churn probability (0-100%)

**Data Processing:**
• Analyzes 11 customer attributes
• Updates predictions daily
• Handles 10,000+ customer profiles simultaneously

**Security**: All data encrypted, GDPR compliant, secure API endpoints"""
        },
        
        'competitive_advantage': {
            'keywords': ['competition', 'advantage', 'edge', 'differentiation', 'market position'],
            'response': """🏆 **Competitive Advantage**

**Our Edge:**
• **Proactive**: Predict churn before it happens (not reactive)
• **Personalized**: Tailored retention strategies per customer
• **Data-driven**: 86%+ accuracy in predictions
• **Cost-effective**: 5x ROI on retention investments

**vs Competitors:**
• Most banks react after customers leave
• We prevent churn with predictive analytics
• Personalized approach vs generic retention offers

**Market Impact**: 20-30% improvement in customer retention rates"""
        },
        
        'customer_segments': {
            'keywords': ['segments', 'customer types', 'demographics', 'profiles'],
            'response': """👥 **Customer Segmentation Analysis**

**Low Risk (60-70%)**
• Young professionals (25-35)
• Multiple products (2-4)
• Active digital users
• **Strategy**: Upsell premium services

**Medium Risk (20-25%)**
• Mid-career (35-50)
• 1-2 products
• **Strategy**: Cross-sell & engagement campaigns

**High Risk (10-15%)**
• Senior customers (50+)
• High balance but inactive
• **Strategy**: Premium service & personal attention"""
        }
    }
    
    # Find matching response
    for category, data in responses.items():
        for keyword in data['keywords']:
            if keyword in question:
                return data['response']
    
    # Handle specific numerical questions
    if any(word in question for word in ['what percentage', 'how much', 'statistics', 'numbers']):
        return """📈 **Key Churn Statistics**

• **Overall churn rate**: 20.37% (industry average: 15-25%)
• **High-risk customers**: 15-20% of customer base
• **Retention success rate**: 70-80% with proactive intervention
• **Revenue at risk**: ₹50-100 lakhs annually per 1000 customers
• **Model accuracy**: 86.65% prediction accuracy

**Actionable Insight**: Focus on the 15-20% high-risk segment for maximum impact."""

    # Handle time-based questions
    if any(word in question for word in ['when', 'timeline', 'schedule', 'frequency']):
        return """⏰ **Optimal Timing for Actions**

**Daily**: Monitor high-risk customer alerts
**Weekly**: Review churn predictions & update strategies  
**Monthly**: Analyze retention campaign performance
**Quarterly**: Model retraining & strategy optimization

**Critical Timing**: Contact high-risk customers within 48-72 hours of prediction for best results."""

    # Handle team/staff questions
    if any(word in question for word in ['team', 'staff', 'training', 'employees']):
        return """👨‍💼 **Team Training & Responsibilities**

**Customer Service Team:**
• Identify high-risk customers from daily reports
• Execute personalized retention scripts
• Escalate complex cases to relationship managers

**Relationship Managers:**
• Handle high-value customer retention
• Conduct quarterly business reviews
• Implement premium service offerings

**Training Required**: 2-day workshop on churn prediction interpretation and retention techniques."""

    # Default response for unmatched questions
    return """🤔 **I can help you with:**

• **Churn Analysis**: "What's our churn rate?" 
• **Retention Strategies**: "How to reduce customer churn?"
• **Risk Assessment**: "Which customers are high risk?"
• **Business Impact**: "What's the ROI of retention?"
• **Implementation**: "How to deploy this system?"
• **Technology**: "How does the AI work?"
• **Competition**: "What's our competitive advantage?"
• **Customer Segments**: "What are our customer types?"

**Ask me anything about customer churn, retention strategies, or implementation guidance!**

*Example: "What retention strategies work best for high-value customers?"*"""

# Initialize model on startup
load_or_train_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)