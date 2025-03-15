// script.js
async function analyzeNews() {
    const newsInput = document.getElementById("news-input").value;
    const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ news: newsInput }),
    });
    
    const result = await response.json();
    document.getElementById("result").innerText = JSON.stringify(result, null, 2);
}
