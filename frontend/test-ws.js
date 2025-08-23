// Simple WebSocket connection test
const ws = new WebSocket('ws://localhost:8001/ws/messagebus');

ws.onopen = function(event) {
    console.log('✅ WebSocket connection opened successfully!');
    console.log('Connection event:', event);
    
    // Send a test message
    ws.send(JSON.stringify({
        type: 'ping',
        message: 'test connection'
    }));
};

ws.onmessage = function(event) {
    console.log('📨 Received message:', event.data);
};

ws.onerror = function(error) {
    console.error('❌ WebSocket error:', error);
};

ws.onclose = function(event) {
    console.log('🔴 WebSocket closed:', event.code, event.reason);
};

// Keep the script running
setTimeout(() => {
    if (ws.readyState === WebSocket.OPEN) {
        console.log('✅ Connection is stable');
        ws.close();
    } else {
        console.log('❌ Connection failed or closed');
    }
}, 3000);