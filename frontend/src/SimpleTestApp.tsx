import React from 'react';

function SimpleTestApp() {
  return (
    <div style={{ 
      padding: '40px',
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f5f5f5',
      minHeight: '100vh'
    }}>
      <h1 style={{ color: '#333', fontSize: '32px' }}>
        🚀 Nautilus Trading Platform
      </h1>
      
      <div style={{
        backgroundColor: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginTop: '20px'
      }}>
        <h2 style={{ color: '#52c41a' }}>✅ React App Working!</h2>
        <p>This is a minimal React component without complex dependencies.</p>
        
        <div style={{ marginTop: '20px' }}>
          <p><strong>Backend:</strong> http://localhost:8000 ✅</p>
          <p><strong>Frontend:</strong> http://localhost:3001 ✅</p>
          <p><strong>Status:</strong> Ready for Trading 🎯</p>
        </div>
        
        <button 
          onClick={() => alert('React onClick events work!')}
          style={{
            backgroundColor: '#1890ff',
            color: 'white',
            border: 'none',
            padding: '10px 20px',
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '20px'
          }}
        >
          Test React Event Handling
        </button>
      </div>
    </div>
  );
}

export default SimpleTestApp;