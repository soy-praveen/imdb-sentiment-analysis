document.getElementById('analyzeBtn').addEventListener('click', function() {
    const review = document.getElementById('reviewInput').value;

    if (!review) {
        alert("Please enter a review!");
        return;
    }

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ review })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Sentiment: ${data.sentiment} | Accuracy: ${data.accuracy}`;
    })
    .catch(error => console.error('Error:', error));
});
