import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Individual form components for better organization
const FormField = ({ label, name, type, value, onChange, placeholder, required, options }) => {
  if (type === 'select') {
    return (
      <div className="form-field">
        <label htmlFor={name}>{label} {required && <span className="required">*</span>}</label>
        <select name={name} value={value} onChange={onChange} required={required}>
          {options.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </div>
    );
  }

  if (type === 'checkbox') {
    return (
      <div className="form-field checkbox-field">
        <label className="checkbox-label">
          <input
            type="checkbox"
            name={name}
            checked={value}
            onChange={onChange}
          />
          <span className="checkmark"></span>
          {label}
        </label>
      </div>
    );
  }

  return (
    <div className="form-field">
      <label htmlFor={name}>{label} {required && <span className="required">*</span>}</label>
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
      />
    </div>
  );
};

const ResultCard = ({ result }) => {
  if (!result) return null;

  const riskPercentage = (result.churn_probability * 100).toFixed(1);
  const confidencePercentage = (result.confidence * 100).toFixed(1);

  return (
    <div className="result-card">
      <div className="result-header">
        <div className="risk-indicator" style={{ backgroundColor: result.risk_color }}>
          <span className="risk-level">{result.risk_level} RISK</span>
          <span className="risk-percentage">{riskPercentage}%</span>
        </div>
        <div className="confidence-score">
          Confidence: {confidencePercentage}%
        </div>
      </div>
      
      <div className="customer-summary">
        <h3>Customer Profile Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="summary-label">Credit Score:</span>
            <span className={`summary-value ${result.customer_summary.credit_score_category.toLowerCase()}`}>
              {result.customer_summary.credit_score_category}
            </span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Tenure:</span>
            <span className="summary-value">{result.customer_summary.tenure_years} years</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Product Usage:</span>
            <span className="summary-value">{result.customer_summary.product_usage}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Engagement:</span>
            <span className={`summary-value ${result.customer_summary.engagement_level.toLowerCase()}`}>
              {result.customer_summary.engagement_level}
            </span>
          </div>
        </div>
      </div>

      <div className="retention-strategies">
        <h3>Recommended Retention Strategies</h3>
        <ul className="strategies-list">
          {result.retention_strategies.map((strategy, index) => (
            <li key={index} className="strategy-item">{strategy}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

const ManagerChatbot = ({ isOpen, onToggle }) => {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: `🏦 **Hello! I'm your AI Assistant for Customer Churn Analysis**

I can help you with:
• Customer retention strategies
• Churn analysis insights  
• Implementation guidance
• Business impact calculations
• Risk assessment

**Ask me anything!** *Example: "What's the best way to retain high-value customers?"*`,
      timestamp: new Date()
    }
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage = {
      type: 'user',
      content: currentMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/chatbot', {
        question: currentMessage
      });

      const botMessage = {
        type: 'bot',
        content: response.data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'bot',
        content: "Sorry, I'm having trouble connecting to the server. Please make sure the Flask server is running and try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (content) => {
    // Convert markdown-style formatting to HTML
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/•/g, '▸')
      .split('\n')
      .map((line, index) => (
        <div key={index} dangerouslySetInnerHTML={{ __html: line }} />
      ));
  };

  const quickQuestions = [
    "What's our churn rate?",
    "How to reduce customer churn?",
    "Which customers are high risk?",
    "What's the ROI of retention?",
    "How to implement this system?"
  ];

  return (
    <>
      <div className={`chatbot-container ${isOpen ? 'open' : 'closed'}`}>
        <div className="chatbot-header">
          <div className="chatbot-title">
            <span className="bot-icon">🤖</span>
            <div>
              <h3>Manager AI Assistant</h3>
              <p>Churn Analysis Expert</p>
            </div>
          </div>
          <button onClick={onToggle} className="close-chat">×</button>
        </div>

        <div className="chatbot-messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-content">
                {formatMessage(message.content)}
              </div>
              <div className="message-time">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
        </div>

        <div className="quick-questions">
          <p>Quick questions:</p>
          <div className="question-buttons">
            {quickQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => {
                  setCurrentMessage(question);
                  setTimeout(() => sendMessage(), 100);
                }}
                className="quick-question"
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        <div className="chatbot-input">
          <textarea
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about churn analysis, retention strategies, or implementation..."
            disabled={isLoading}
            rows="2"
          />
          <button onClick={sendMessage} disabled={isLoading || !currentMessage.trim()}>
            {isLoading ? '⏳' : '📤'}
          </button>
        </div>
      </div>

      <button className="chatbot-toggle" onClick={onToggle}>
        {isOpen ? '✕' : '💬'}
        <span>Manager Assistant</span>
      </button>
    </>
  );
};

function App() {
  const [formData, setFormData] = useState({
    credit_score: '',
    age: '',
    tenure: '',
    balance: '',
    num_of_products: '1',
    estimated_salary: '',
    geography: 'France',
    gender: 'Male',
    has_cr_card: false,
    is_active_member: false,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [serverHealth, setServerHealth] = useState(null);
  const [chatbotOpen, setChatbotOpen] = useState(false);

  // Check server health on component mount
  useEffect(() => {
    checkServerHealth();
  }, []);

  const checkServerHealth = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/health');
      setServerHealth(response.data);
    } catch (err) {
      setServerHealth({ status: 'unhealthy', model_loaded: false });
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const validateForm = () => {
    const { credit_score, age, balance, estimated_salary } = formData;
    
    if (credit_score < 300 || credit_score > 850) {
      throw new Error('Credit score must be between 300 and 850');
    }
    if (age < 18 || age > 100) {
      throw new Error('Age must be between 18 and 100');
    }
    if (balance < 0) {
      throw new Error('Balance cannot be negative');
    }
    if (estimated_salary < 0) {
      throw new Error('Salary cannot be negative');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError('');

    try {
      validateForm();
      
      // Convert boolean values to integers for the API
      const apiData = {
        ...formData,
        has_cr_card: formData.has_cr_card ? 1 : 0,
        is_active_member: formData.is_active_member ? 1 : 0,
      };

      const response = await axios.post('http://127.0.0.1:5000/predict', apiData);
      setResult(response.data);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(`Server Error: ${err.response.data.error}`);
      } else if (err.message) {
        setError(err.message);
      } else {
        setError('Unable to connect to prediction service. Please ensure the Flask server is running on port 5000.');
      }
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = () => {
    setFormData({
      credit_score: '650',
      age: '35',
      tenure: '5',
      balance: '80000',
      num_of_products: '2',
      estimated_salary: '75000',
      geography: 'Germany',
      gender: 'Female',
      has_cr_card: true,
      is_active_member: false,
    });
  };

  const clearForm = () => {
    setFormData({
      credit_score: '',
      age: '',
      tenure: '',
      balance: '',
      num_of_products: '1',
      estimated_salary: '',
      geography: 'France',
      gender: 'Male',
      has_cr_card: false,
      is_active_member: false,
    });
    setResult(null);
    setError('');
  };

  return (
    <div className="app">
      <div className="container">
        <header className="app-header">
          <div className="header-content">
            <h1>🏦 Retail Banking Intelligence Engine (RBIE) </h1>
            <p>Advanced ML-powered customer retention analytics</p>
            {serverHealth && (
              <div className={`server-status ${serverHealth.status}`}>
                <span className="status-indicator"></span>
                Server: {serverHealth.status} | Model: {serverHealth.model_loaded ? 'Loaded' : 'Not Available'}
              </div>
            )}
          </div>
        </header>

        <div className="main-content">
          <div className="form-section">
            <div className="form-header">
              <h2>Customer Information</h2>
              <div className="form-actions">
                <button type="button" onClick={loadSampleData} className="btn-secondary">
                  Load Sample Data
                </button>
                <button type="button" onClick={clearForm} className="btn-secondary">
                  Clear Form
                </button>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="prediction-form">
              <div className="form-grid">
                <FormField
                  label="Credit Score"
                  name="credit_score"
                  type="number"
                  value={formData.credit_score}
                  onChange={handleChange}
                  placeholder="300-850"
                  required
                />
                <FormField
                  label="Age"
                  name="age"
                  type="number"
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="18-100"
                  required
                />
                <FormField
                  label="Tenure (Years)"
                  name="tenure"
                  type="number"
                  value={formData.tenure}
                  onChange={handleChange}
                  placeholder="0-10"
                  required
                />
                <FormField
                  label="Account Balance (₹)"
                  name="balance"
                  type="number"
                  value={formData.balance}
                  onChange={handleChange}
                  placeholder="0"
                  required
                />
                <FormField
                  label="Number of Products"
                  name="num_of_products"
                  type="select"
                  value={formData.num_of_products}
                  onChange={handleChange}
                  required
                  options={[
                    { value: '1', label: '1 Product' },
                    { value: '2', label: '2 Products' },
                    { value: '3', label: '3 Products' },
                    { value: '4', label: '4+ Products' }
                  ]}
                />
                <FormField
                  label="Estimated Salary (₹)"
                  name="estimated_salary"
                  type="number"
                  value={formData.estimated_salary}
                  onChange={handleChange}
                  placeholder="Annual salary"
                  required
                />
                <FormField
                  label="Geography"
                  name="geography"
                  type="select"
                  value={formData.geography}
                  onChange={handleChange}
                  required
                  options={[
                    { value: 'France', label: 'France' },
                    { value: 'Spain', label: 'Spain' },
                    { value: 'Germany', label: 'Germany' }
                  ]}
                />
                <FormField
                  label="Gender"
                  name="gender"
                  type="select"
                  value={formData.gender}
                  onChange={handleChange}
                  required
                  options={[
                    { value: 'Male', label: 'Male' },
                    { value: 'Female', label: 'Female' }
                  ]}
                />
              </div>

              <div className="checkbox-section">
                <FormField
                  label="Has Credit Card"
                  name="has_cr_card"
                  type="checkbox"
                  value={formData.has_cr_card}
                  onChange={handleChange}
                />
                <FormField
                  label="Is Active Member"
                  name="is_active_member"
                  type="checkbox"
                  value={formData.is_active_member}
                  onChange={handleChange}
                />
              </div>

              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing Customer Profile...
                  </>
                ) : (
                  <>
                    🔍 Predict Churn Risk
                  </>
                )}
              </button>
            </form>
          </div>

          <div className="results-section">
            {error && (
              <div className="error-card">
                <h3>⚠️ Error</h3>
                <p>{error}</p>
                {error.includes('Flask server') && (
                  <div className="error-help">
                    <p><strong>To fix this:</strong></p>
                    <ol>
                      <li>Make sure you have Flask installed: <code>pip install flask flask-cors joblib pandas scikit-learn</code></li>
                      <li>Run the Flask server: <code>python app.py</code></li>
                      <li>Ensure the server is running on http://127.0.0.1:5000</li>
                    </ol>
                  </div>
                )}
              </div>
            )}
            
            <ResultCard result={result} />
          </div>
        </div>
      </div>

      {/* Manager Chatbot */}
      <ManagerChatbot 
        isOpen={chatbotOpen} 
        onToggle={() => setChatbotOpen(!chatbotOpen)} 
      />
    </div>
  );
}

export default App;