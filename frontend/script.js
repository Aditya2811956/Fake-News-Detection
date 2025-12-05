const API_URL = 'http://localhost:5001';

async function analyzeNews() {
    const newsText = document.getElementById('newsText').value.trim();
    
    if (!newsText) {
        alert('Please enter some news text to analyze!');
        return;
    }
    
    // Hide result section and show loading
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'block';
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: newsText })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
        } else {
            alert('Error: ' + (data.error || 'Something went wrong'));
        }
    } catch (error) {
        alert('Failed to connect to the server. Make sure the backend is running on port 5000.');
        console.error('Error:', error);
    } finally {
        document.getElementById('loadingSpinner').style.display = 'none';
    }
}

function displayResult(data) {
    // Show result section
    document.getElementById('resultSection').style.display = 'block';
    
    // Update prediction badge
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionText = document.getElementById('predictionText');
    
    predictionText.textContent = data.prediction;
    
    if (data.prediction === 'FAKE') {
        predictionBadge.className = 'prediction-badge fake';
    } else {
        predictionBadge.className = 'prediction-badge real';
    }
    
    // Update confidence meter
    const progressFill = document.getElementById('progressFill');
    const confidenceText = document.getElementById('confidenceText');
    
    progressFill.style.width = data.confidence + '%';
    confidenceText.textContent = data.confidence + '%';
    
    // Update message
    document.getElementById('resultMessage').textContent = data.message;
    
    // Scroll to result
    document.getElementById('resultSection').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}

function clearForm() {
    document.getElementById('newsText').value = '';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
}

// Allow Enter key to submit (with Ctrl/Cmd)
document.getElementById('newsText').addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        analyzeNews();
    }
});
