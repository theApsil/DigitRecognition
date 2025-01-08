const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const result = document.getElementById('result');

let painting = false;

canvas.addEventListener('mousedown', () => (painting = true));
canvas.addEventListener('mouseup', () => (painting = false));
canvas.addEventListener('mousemove', (e) => {
    if (!painting) return;
    ctx.fillStyle = 'black';
    ctx.fillRect(e.offsetX, e.offsetY, 10, 10);
});

document.getElementById('predict').addEventListener('click', async () => {
    const data = canvas.toDataURL('image/png');
    const response = await fetch('/recognize', {
        method: 'POST',
        body: data
    });
    const json = await response.json();
    result.textContent = `Result: ${json.digit}`;
});
