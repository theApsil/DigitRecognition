const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', () => (isDrawing = true));
canvas.addEventListener('mouseup', () => (isDrawing = false));
canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!isDrawing) return;
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(event.offsetX, event.offsetY, 5, 0, Math.PI * 2);
    ctx.fill();
}

document.getElementById('predict').addEventListener('click', async () => {
    const data = canvas.toDataURL('image/png');
    const res = await fetch('/predict', {
        method: 'POST',
        body: data,
    });
    const result = await res.json();
    document.getElementById('result').textContent = result.success
        ? `Prediction: ${result.prediction}`
        : `Error: ${result.error}`;
});
