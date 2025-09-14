const socket = new WebSocket("ws://localhost:8000/");
const canvas = document.getElementById("lineplot");
const ctx = canvas.getContext("2d");
let points = [];
let collecting = false;

document.getElementById("showCanvasBtn").onclick = function() {
    document.getElementById("lineplot").style.display = "block";
    collecting = true
};


socket.onopen = function(event) {
    console.log("WebSocket connected!");
    
};

socket.onmessage = function(event) {
    if (collecting == true) {
        console.log(event.data);
        const data = JSON.parse(event.data);
        currentX = (data.x / 900) * canvas.width
        drawMovingLine();
    }
    
};

socket.onclose = function(event) {
    console.log("WebSocket closed.");
};

socket.onerror = function(error) {
    console.error("WebSocket error:", error);
};

function drawMovingLine() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, canvas.height);
    ctx.stroke();
}