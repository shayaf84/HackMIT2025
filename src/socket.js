const socket = new WebSocket("ws://localhost:8000/");
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
let points = [];
let collecting = false;

document.getElementById("showCanvasBtn").onclick = function() {
    document.getElementById("plot").style.display = "block";
    collecting = true
};


socket.onopen = function(event) {
    console.log("WebSocket connected!");
    
};

socket.onmessage = function(event) {
    if (collecting == true) {
        console.log(event.data);
        const data = JSON.parse(event.data);
        points.push([data.x, data.y]);
        drawPoints();
    }
    
};

socket.onclose = function(event) {
    console.log("WebSocket closed.");
};

socket.onerror = function(error) {
    console.error("WebSocket error:", error);
};

function drawPoints() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "red";
    for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2*Math.PI);
        ctx.fill();
    }
}