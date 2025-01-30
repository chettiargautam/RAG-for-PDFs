document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();
    
    const formData = new FormData(this);
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => alert(data.message))
    .catch(error => console.error('Error:', error));
});

function askQuestion() {
    const query = document.getElementById('userQuery').value;
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => displayResponse(data))
    .catch(error => console.error('Error:', error));
}

function displayResponse(results) {
    const chatWindow = document.getElementById('chatWindow');
    results.forEach(result => {
        const messageDiv = document.createElement('div');
        messageDiv.innerHTML = `<strong>Answer:</strong> ${result.answer}<br>
                               <strong>From File:</strong> ${result.file}<br>
                               <strong>Chunk Number:</strong> ${result.chunk_number}<br>
                               <strong>Context:</strong> ${result.retrieved_context}`;
        chatWindow.appendChild(messageDiv);
    });
}
