// popup.js

document.addEventListener("DOMContentLoaded", () => {
  const API_URL = 'http://localhost:5001';  // Your Flask app URL
  const predictBtn = document.getElementById("predictBtn");
  const resultsDiv = document.getElementById("results");
  
  // Form elements
  const formElements = {
    age: document.getElementById("age"),
    sex: document.getElementById("sex"),
    chestPainType: document.getElementById("chestPainType"),
    restingBP: document.getElementById("restingBP"),
    cholesterol: document.getElementById("cholesterol"),
    fastingBS: document.getElementById("fastingBS"),
    restingECG: document.getElementById("restingECG"),
    maxHR: document.getElementById("maxHR"),
    exerciseAngina: document.getElementById("exerciseAngina"),
    oldpeak: document.getElementById("oldpeak"),
    stSlope: document.getElementById("stSlope")
  };

  // Feature name mapping for the API
  const featureMapping = {
    age: 'Age',
    sex: 'Sex',
    chestPainType: 'ChestPainType',
    restingBP: 'RestingBP',
    cholesterol: 'Cholesterol',
    fastingBS: 'FastingBS',
    restingECG: 'RestingECG',
    maxHR: 'MaxHR',
    exerciseAngina: 'ExerciseAngina',
    oldpeak: 'Oldpeak',
    stSlope: 'ST_Slope'
  };

  // Validation rules
  const validationRules = {
    age: { min: 1, max: 120, required: true },
    sex: { required: true },
    chestPainType: { required: true },
    restingBP: { min: 50, max: 250, required: true },
    cholesterol: { min: 100, max: 600, required: true },
    fastingBS: { required: true },
    restingECG: { required: true },
    maxHR: { min: 60, max: 220, required: true },
    exerciseAngina: { required: true },
    oldpeak: { min: 0, max: 10, required: true },
    stSlope: { required: true }
  };

  // Validate form data
  function validateForm() {
    const errors = [];
    
    for (const [fieldName, element] of Object.entries(formElements)) {
      const value = element.value.trim();
      const rules = validationRules[fieldName];
      
      // Check if required field is empty
      if (rules.required && !value) {
        errors.push(`${getFieldDisplayName(fieldName)} is required`);
        continue;
      }
      
      // Check numeric ranges
      if (value && rules.min !== undefined && parseFloat(value) < rules.min) {
        errors.push(`${getFieldDisplayName(fieldName)} must be at least ${rules.min}`);
      }
      
      if (value && rules.max !== undefined && parseFloat(value) > rules.max) {
        errors.push(`${getFieldDisplayName(fieldName)} must be at most ${rules.max}`);
      }
    }
    
    return errors;
  }

  // Get user-friendly field names for error messages
  function getFieldDisplayName(fieldName) {
    const displayNames = {
      age: 'Age',
      sex: 'Sex',
      chestPainType: 'Chest Pain Type',
      restingBP: 'Resting Blood Pressure',
      cholesterol: 'Cholesterol',
      fastingBS: 'Fasting Blood Sugar',
      restingECG: 'Resting ECG',
      maxHR: 'Maximum Heart Rate',
      exerciseAngina: 'Exercise Induced Angina',
      oldpeak: 'ST Depression (Oldpeak)',
      stSlope: 'ST Slope'
    };
    return displayNames[fieldName] || fieldName;
  }

  // Collect form data
  function collectFormData() {
    const data = {};
    
    for (const [fieldName, element] of Object.entries(formElements)) {
      const apiFieldName = featureMapping[fieldName];
      let value = element.value.trim();
      
      // Convert to appropriate data type
      if (fieldName === 'oldpeak') {
        value = parseFloat(value);
      } else {
        value = parseInt(value);
      }
      
      data[apiFieldName] = value;
    }
    
    return data;
  }

  // Display loading state
  function showLoading() {
    resultsDiv.style.display = "block";
    resultsDiv.innerHTML = `
      <div class="loading">
        <p>🔄 Analyzing health data...</p>
        <p>Please wait while we process your information.</p>
      </div>
    `;
  }

  // Display error message
  function showError(message) {
    resultsDiv.style.display = "block";
    resultsDiv.innerHTML = `
      <div class="error">
        <p><strong>❌ Error:</strong> ${message}</p>
      </div>
    `;
  }

  // Display validation errors
  function showValidationErrors(errors) {
    const errorList = errors.map(error => `<li>${error}</li>`).join('');
    resultsDiv.style.display = "block";
    resultsDiv.innerHTML = `
      <div class="error">
        <p><strong>❌ Please fix the following errors:</strong></p>
        <ul>${errorList}</ul>
      </div>
    `;
  }

  // Display prediction results
  function showResults(data) {
    const riskClass = data.prediction === 0 ? 'low-risk' : 'high-risk';
    const riskIcon = data.prediction === 0 ? '✅' : '⚠️';
    
    resultsDiv.style.display = "block";
    resultsDiv.innerHTML = `
      <div class="results-section">
        <div class="section-title">Prediction Results</div>
        
        <div class="risk-indicator ${riskClass}">
          ${riskIcon} ${data.risk_level}
        </div>
        
        <p style="text-align: center; margin: 15px 0; color: #cccccc;">
          ${data.message}
        </p>
        
        <div class="confidence-bars">
          <div class="section-title" style="margin-bottom: 10px;">Confidence Levels</div>
          
          <div class="confidence-bar">
            <span class="confidence-label">Low Risk:</span>
            <div class="confidence-progress">
              <div class="confidence-fill" style="width: ${data.confidence.low_risk}%; background-color: #4caf50;"></div>
            </div>
            <span class="confidence-value">${data.confidence.low_risk}%</span>
          </div>
          
          <div class="confidence-bar">
            <span class="confidence-label">High Risk:</span>
            <div class="confidence-progress">
              <div class="confidence-fill" style="width: ${data.confidence.high_risk}%; background-color: #f44336;"></div>
            </div>
            <span class="confidence-value">${data.confidence.high_risk}%</span>
          </div>
        </div>
        
        <div class="chart-container">
          <button id="generateChart" style="
            background-color: #ff6b6b;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
          ">📊 Generate Risk Chart</button>
        </div>
      </div>
    `;
    
    // Add event listener for chart generation
    document.getElementById("generateChart").addEventListener("click", () => {
      generateRiskChart([data.prediction]);
    });
  }

  // Generate and display risk chart
  async function generateRiskChart(predictions) {
    try {
      const response = await fetch(`${API_URL}/generate_risk_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ predictions })
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate chart');
      }
      
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      
      // Replace the button with the chart
      const chartContainer = document.querySelector('.chart-container');
      chartContainer.innerHTML = `
        <img src="${imgURL}" alt="Risk Distribution Chart" style="max-width: 100%; height: auto; border-radius: 6px; margin-top: 10px;">
      `;
      
    } catch (error) {
      console.error("Error generating chart:", error);
      const chartContainer = document.querySelector('.chart-container');
      chartContainer.innerHTML = `
        <p style="color: #ff9090; font-size: 12px; margin-top: 10px;">
          ❌ Could not generate chart
        </p>
      `;
    }
  }

  // Make prediction API call
  async function makePrediction(patientData) {
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_data: patientData })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }
      
      const result = await response.json();
      return result;
      
    } catch (error) {
      console.error("Prediction error:", error);
      throw error;
    }
  }

  // Main prediction handler
  async function handlePrediction() {
    // Validate form
    const validationErrors = validateForm();
    if (validationErrors.length > 0) {
      showValidationErrors(validationErrors);
      return;
    }
    
    // Show loading state
    showLoading();
    predictBtn.disabled = true;
    predictBtn.textContent = "Processing...";
    
    try {
      // Collect and send data
      const patientData = collectFormData();
      const result = await makePrediction(patientData);
      
      // Show results
      showResults(result);
      
    } catch (error) {
      showError(error.message || 'An unexpected error occurred');
    } finally {
      // Reset button
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict Risk";
    }
  }

  // Test API connection on load
  async function testConnection() {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (!response.ok) {
        throw new Error('API not responding');
      }
      console.log("✅ API connection successful");
    } catch (error) {
      console.error("❌ API connection failed:", error);
      resultsDiv.style.display = "block";
      resultsDiv.innerHTML = `
        <div class="error">
          <p><strong>⚠️ Connection Error:</strong> Cannot connect to prediction service. Please ensure the Flask app is running on ${API_URL}</p>
        </div>
      `;
    }
  }

  // Event listeners
  predictBtn.addEventListener("click", handlePrediction);
  
  // Test connection on load
  testConnection();
  
  // Add input validation listeners
  Object.entries(formElements).forEach(([fieldName, element]) => {
    element.addEventListener("input", () => {
      // Clear previous error styling
      element.style.borderColor = "#555";
    });
  });
});